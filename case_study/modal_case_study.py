"""Run the book's case-study experiments on several GPUs with Modal and check the book's principles.

Two kinds of content are kept apart on purpose:
  * experiment_config.py: what is swappable (GPUs, models, sizes, package versions).
  * principles.py: what the book teaches and must keep holding on every GPU.

For each GPU it runs, in parallel:
  1. chapter02/chapter02_lab.ipynb, executed top to bottom (generator + embedder + FAISS): latency
     distribution, stage table, profiler table, Roofline placement.
  2. Extra experiments: embedder throughput, precision and attention kernel (Chapter 3); compiled
     decode, wrong vs right torch.compile pattern (Chapter 4); prefill and decode against batch
     size (Chapter 5).
Then it checks every principle against every GPU's results.

Prerequisites (once):  pip install modal && modal setup
                       modal secret create huggingface HF_TOKEN=<read token>   (accept the Gemma license first)
Run from the repository root:
    modal run case_study/modal_case_study.py                   # every GPU in experiment_config.GPUS
    modal run case_study/modal_case_study.py --gpus T4,H100    # a subset
Outputs in case_study/results/:
    <gpu>.json                       all measurements for one GPU, with software versions
    chapter02_lab_executed_<gpu>.ipynb
    principles_report.md             each principle, holds or fails, per GPU, with evidence
    principles_report.json
"""
import json
import os
import pathlib
import sys

import modal

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))
import experiment_config as cfg  # noqa: E402

hf_cache = modal.Volume.from_name("hpml-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*cfg.PIP_PACKAGES)
    .run_commands("python -m ipykernel install --name python3 --display-name 'Python 3'")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "MPLBACKEND": "Agg"})
    .add_local_file(str(HERE / "experiment_config.py"), "/root/experiment_config.py")
    .add_local_file(str(HERE / "experiments.py"), "/root/experiments.py")
    .add_local_dir(str(REPO / "chapter02"), remote_path="/repo/chapter02")
)

app = modal.App("hpml-case-study", image=image)
# The GPU is a decoration-time argument and this Modal build has no
# Function.with_options(gpu=...) to override it per call, so the functions are
# registered once per GPU below (see the factory after their definitions).
COMMON = dict(timeout=3 * 3600, volumes={"/cache": hf_cache},
              secrets=[modal.Secret.from_name("huggingface")])


def _tag(gpu: str) -> str:
    return gpu.lower().replace("-", "_")


# ------------------------------------------------------------------------------------------------
# helpers (run inside the GPU container)
# ------------------------------------------------------------------------------------------------
def _env():
    import platform
    import subprocess

    import sentence_transformers
    import torch
    import transformers
    name = torch.cuda.get_device_name(0)
    spec = next(({"label": lab, "fp16_tensor_tflops": tf, "mem_bw_gbs": bw, "memory": mem}
                 for key, lab, tf, bw, mem in cfg.GPU_SPECS if key in name), None)
    return {
        "gpu": name,
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "driver": subprocess.getoutput("nvidia-smi --query-gpu=driver_version --format=csv,noheader").strip(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__, "sentence_transformers": sentence_transformers.__version__,
        "python": platform.python_version(),
        # Host CPU matters: at batch 1 decode is often limited by Python and kernel-launch overhead.
        "cpu": next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), "?"),
        "cpu_count": os.cpu_count(),
        "packages": {ln.split("==")[0].lower(): ln.split("==")[1] for ln in
                     subprocess.getoutput(f"{sys.executable} -m pip freeze").splitlines() if "==" in ln},
    }, spec


# The GPU measurement helpers (_cuda_time, _wikitext, _generator, _prompt, ...) live in
# experiments.py alongside the experiments that use them, so they can run in an isolated child
# interpreter. This runner keeps only _env, which it uses to stamp each result with its platform.


# ------------------------------------------------------------------------------------------------
# 1. the Chapter 2 lab, executed headless
# ------------------------------------------------------------------------------------------------
SUMMARY_CELL = r'''
import json as _json, numpy as _np
_out = {"latency_ms": {"p50": float(_np.percentile(totals, 50)), "p95": float(_np.percentile(totals, 95)),
                       "p99": float(_np.percentile(totals, 99)), "mean": float(totals.mean()), "n": int(len(totals))},
        "stage_median_ms": {k: float(v) for k, v in summary.items()}}
_g = globals()
if all(k in _g for k in ("t_per_token", "achieved_flops", "achieved_bw", "peak_bw")):
    _out["decode_roofline"] = {"ms_per_token": float(t_per_token * 1000), "achieved_tflops": float(achieved_flops / 1e12),
                               "achieved_bw_gbs": float(achieved_bw / 1e9), "pct_peak_bw": float(100 * achieved_bw / peak_bw),
                               "arithmetic_intensity": float(_g.get("ai", float("nan"))),
                               "ridge": float(_g.get("ridge_ai", float("nan")))}
_json.dump(_out, open("/tmp/ch2_summary.json", "w"))
'''


def run_chapter02_lab() -> dict:
    import nbformat
    from nbclient import NotebookClient

    os.chdir("/repo/chapter02")
    nb = nbformat.read("chapter02_lab.ipynb", as_version=4)
    for cell in nb.cells:  # libraries are already in the image; skip in-notebook installs
        if cell.cell_type == "code":
            cell.source = "\n".join(("# [skipped on Modal] " + ln) if ln.lstrip().startswith(("!", "%pip")) else ln
                                    for ln in cell.source.split("\n"))
    nb.cells.append(nbformat.v4.new_code_cell(SUMMARY_CELL))
    NotebookClient(nb, timeout=3600, kernel_name="python3", allow_errors=True,
                   resources={"metadata": {"path": "/repo/chapter02"}}).execute()
    errors = [{"cell": i, "ename": o.get("ename"), "evalue": o.get("evalue", "")[:500]}
              for i, c in enumerate(nb.cells) for o in c.get("outputs", []) if o.get("output_type") == "error"]
    summary = json.load(open("/tmp/ch2_summary.json")) if os.path.exists("/tmp/ch2_summary.json") else None
    nb.cells.pop()  # keep the saved notebook identical to the repo's apart from outputs
    hf_cache.commit()
    return {"executed_notebook": nbformat.writes(nb), "errors": errors, "summary": summary}


# ------------------------------------------------------------------------------------------------
# 2. extra experiments
# ------------------------------------------------------------------------------------------------
# The measurements themselves live in experiments.py, which imports no modal. That lets each one
# run in a fresh child interpreter here: on some GPUs the Chapter 4 compiled-decode experiment
# crashes the process natively (SIGSEGV from torch.compile CUDA graphs + the static KV cache), which
# a try/except cannot catch. A subprocess turns that crash into a non-zero exit we record as an
# error, so the embedder and decode-batching numbers for the same GPU still come back.
def _run_experiment_subprocess(name: str) -> dict:
    import json as _json
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name
    proc = subprocess.run([sys.executable, "/root/experiments.py", name, out_path],
                          capture_output=True, text=True)
    partial = None
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        try:
            partial = _json.load(open(out_path))
        except Exception:  # noqa: BLE001
            partial = None
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        err = {"error": f"experiment subprocess exited with code {proc.returncode} "
                        f"(signal {-proc.returncode} if negative)", "stderr_tail": tail}
        # Keep whatever finished before the crash (compiled_decode flushes after each measurement).
        if isinstance(partial, dict):
            err.update(partial)
        return err
    return partial if partial is not None else {"error": "experiment produced no output"}


def run_experiments() -> dict:
    import torch
    env, spec = _env()
    res = {"env": env, "spec": spec, "config": {k: getattr(cfg, k) for k in
           ("MODELS", "DTYPE", "PROMPT_TOKENS", "NEW_TOKENS", "DECODE_BATCHES", "EMBED_BATCHES", "TIMING")}}
    for key in ("embedder", "decode_batching"):
        res[key] = _run_experiment_subprocess(key)
        torch.cuda.empty_cache()
    # Chapter 4: each decode variant in its own child process, merged into one "compiled_decode" dict.
    cd, runs = {}, {}
    for key in ("decode_eager", "decode_wrong_compile", "decode_compile_reduce_overhead",
                "decode_compile_default"):
        r = _run_experiment_subprocess(key)
        runs[key] = {k: v for k, v in r.items() if k in ("error", "stderr_tail", "mode",
                                                       "autocompile_switch_available")}
        if key.startswith("decode_compile_") and "right_compile_forward_static_cache" in r:
            r[f"right_compile_{key.removeprefix('decode_compile_')}"] = r.pop("right_compile_forward_static_cache")
        cd.update({k: v for k, v in r.items() if k not in ("error", "stderr_tail", "mode")})
    # the book's "right" number: the CUDA-graph version if it ran, otherwise the default-mode one
    for k in ("right_compile_reduce_overhead", "right_compile_default"):
        if k in cd:
            cd["right_compile_forward_static_cache"] = {**cd[k], "mode": k.removeprefix("right_compile_")}
            break
    cd["runs"] = runs
    res["compiled_decode"] = cd
    hf_cache.commit()
    return res


# ------------------------------------------------------------------------------------------------
# per-GPU function registration
# ------------------------------------------------------------------------------------------------
# One pair of Modal functions per configured GPU, each pinned to its GPU at decoration time. This
# replaces Function.with_options(gpu=...), which this Modal build no longer provides, and still lets
# every GPU run in parallel from one app.
LAB_FUNCS, EXP_FUNCS = {}, {}
for _gpu in cfg.GPUS:
    _t = _tag(_gpu)
    LAB_FUNCS[_gpu] = app.function(name=f"run_chapter02_lab_{_t}", gpu=_gpu, **COMMON)(run_chapter02_lab)
    EXP_FUNCS[_gpu] = app.function(name=f"run_experiments_{_t}", gpu=_gpu, **COMMON)(run_experiments)


# ------------------------------------------------------------------------------------------------
# local entrypoint
# ------------------------------------------------------------------------------------------------
@app.local_entrypoint()
def main(gpus: str = ""):
    import principles

    RESULTS.mkdir(parents=True, exist_ok=True)
    targets = [g.strip() for g in gpus.split(",") if g.strip()] or list(cfg.GPUS)
    missing = [g for g in targets if g not in LAB_FUNCS]
    if missing:
        raise SystemExit(f"GPUs not in experiment_config.GPUS: {missing}. Add them there first.")
    jobs = {g: (LAB_FUNCS[g].spawn(), EXP_FUNCS[g].spawn()) for g in targets}
    by_gpu = {}
    for g, (lab_job, exp_job) in jobs.items():
        tag = g.lower().replace("-", "_")
        lab, exp = lab_job.get(), exp_job.get()
        (RESULTS / f"chapter02_lab_executed_{tag}.ipynb").write_text(lab.pop("executed_notebook"))
        res = {"requested_gpu": g, **exp, "chapter02_lab": lab}
        (RESULTS / f"{tag}.json").write_text(json.dumps(res, indent=2))
        by_gpu[g] = res
        print(f"{g}: lab errors={len(lab['errors'])}; experiment errors="
              f"{[k for k in ('embedder', 'decode_batching') if 'error' in (res.get(k) or {})]}"
              f"; decode runs with errors={[k for k, v in res['compiled_decode']['runs'].items() if 'error' in v]}")
    rows = principles.evaluate(by_gpu)
    (RESULTS / "principles_report.json").write_text(json.dumps(rows, indent=2))
    (RESULTS / "principles_report.md").write_text(principles.markdown_report(rows, by_gpu))
    print("\n" + principles.markdown_report(rows, by_gpu).split("## Evidence")[0])
    print(f"Results in {RESULTS}")
