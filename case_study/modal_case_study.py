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
    # Nsight Systems CLI for the Chapter 2 trace. If the install fails the image still builds and the
    # nsys step reports "nsys not available" instead of breaking the run.
    .apt_install("wget", "gnupg", "ca-certificates")
    .run_commands(
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb"
        " && dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update"
        " && (apt-get install -y --no-install-recommends nsight-systems-cli"
        " || apt-get install -y --no-install-recommends $(apt-cache search --names-only '^nsight-systems-[0-9]' | sort | tail -1 | cut -d' ' -f1))"
        " || echo 'nsight systems not installed'")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "MPLBACKEND": "Agg"})
    .add_local_file(str(HERE / "experiment_config.py"), "/root/experiment_config.py")
    .add_local_file(str(HERE / "experiments.py"), "/root/experiments.py")
    .add_local_file(str(HERE / "experiments_extra.py"), "/root/experiments_extra.py")
    .add_local_dir(str(REPO / "chapter02"), remote_path="/repo/chapter02")
)

app = modal.App("hpml-case-study", image=image)
# The GPU is a decoration-time argument and this Modal build has no
# Function.with_options(gpu=...) to override it per call, so the functions are
# registered once per GPU below (see the factory after their definitions).
COMMON = dict(timeout=3 * 3600, volumes={"/cache": hf_cache},
              secrets=[modal.Secret.from_name("huggingface")])


def _tag(gpu: str) -> str:
    return gpu.lower().replace("-", "_").replace("!", "")


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
import torch as _torch
_out = {"gpu": _torch.cuda.get_device_name(0), "latency_ms": {"p50": float(_np.percentile(totals, 50)), "p95": float(_np.percentile(totals, 95)),
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


def run_chapter02_lab(with_traces: bool = False) -> dict:
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
    files = {}
    if with_traces:  # the lab writes these two Chrome traces itself (sections on the profiler and NVTX)
        import gzip
        for fn in ("rag_decode_trace.json", "rag_decode_nvtx_trace.json"):
            if os.path.exists(fn):
                files[f"chapter02_lab/{fn}.gz"] = gzip.compress(open(fn, "rb").read())
    hf_cache.commit()
    return {"executed_notebook": nbformat.writes(nb), "errors": errors, "summary": summary, "_files": files}


# ------------------------------------------------------------------------------------------------
# 2. extra experiments
# ------------------------------------------------------------------------------------------------
# The measurements themselves live in experiments.py, which imports no modal. That lets each one
# run in a fresh child interpreter here: on some GPUs the Chapter 4 compiled-decode experiment
# crashes the process natively (SIGSEGV from torch.compile CUDA graphs + the static KV cache), which
# a try/except cannot catch. A subprocess turns that crash into a non-zero exit we record as an
# error, so the embedder and decode-batching numbers for the same GPU still come back.
def _run_experiment_subprocess(name: str, script: str = "/root/experiments.py", env_extra=None,
                               files=None, prefix=None) -> dict:
    """Run one experiment in a child interpreter. Files it writes to EXP_OUT_DIR are added to
    `files` (dict) under `prefix/` when given."""
    import json as _json
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name
    out_dir = tempfile.mkdtemp(prefix=f"exp_{name}_")
    env = {**os.environ, "EXP_OUT_DIR": out_dir, **(env_extra or {})}
    proc = subprocess.run([sys.executable, script, name, out_path], capture_output=True, text=True, env=env)
    if files is not None:
        for root, _d, fs in os.walk(out_dir):
            for fn in fs:
                p = os.path.join(root, fn)
                files[f"{prefix or name}/{os.path.relpath(p, out_dir)}"] = open(p, "rb").read()
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


def _run_nsys(files) -> dict:
    """Nsight Systems capture of prefill + 5 NVTX-labelled decode steps; returns the .nsys-rep and
    CSV summaries (kernels, CUDA API, NVTX ranges)."""
    import glob
    import shutil
    import subprocess
    import tempfile

    nsys = shutil.which("nsys") or next(iter(sorted(glob.glob("/opt/nvidia/nsight-systems*/**/bin/nsys",
                                                              recursive=True))), None) \
        or next(iter(sorted(glob.glob("/usr/local/cuda*/bin/nsys"))), None)
    if not nsys:
        return {"error": "nsys not available in the image"}
    d = tempfile.mkdtemp(prefix="nsys_")
    rep = os.path.join(d, "rag_decode")
    out = {"nsys": subprocess.getoutput(f"{nsys} --version").strip(), "attempts": []}
    target = [sys.executable, "/root/experiments_extra.py", "nsys_target", os.path.join(d, "target.json")]
    for trace in ("cuda,nvtx,osrt,cublas", "cuda,nvtx"):
        cmd = [nsys, "profile", f"--trace={trace}", "--sample=none", "--cpuctxsw=none",
               "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--force-overwrite=true",
               "-o", rep, *target]
        p = subprocess.run(cmd, capture_output=True, text=True)
        ok = p.returncode == 0 and os.path.exists(rep + ".nsys-rep")
        out["attempts"].append({"trace": trace, "returncode": p.returncode, "ok": ok,
                                "tail": (p.stderr or p.stdout or "")[-1500:]})
        if ok:
            out["trace"] = trace
            break
    if not os.path.exists(rep + ".nsys-rep"):
        out["error"] = "nsys capture failed (see attempts)"
        return out
    subprocess.run([nsys, "stats", "--report", "cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum", "--format", "csv",
                    "--output", rep, "--force-export=true", rep + ".nsys-rep"], capture_output=True, text=True)
    for fn in os.listdir(d):
        if fn.endswith((".nsys-rep", ".csv")):
            files[f"nsys/{fn}"] = open(os.path.join(d, fn), "rb").read()
    out["files"] = sorted(k for k in files if k.startswith("nsys/"))
    return out


EXTRA_PER_REPEAT = ["warmup", "python_loop", "autograd", "dataloader", "compile_stages", "gpt2_explain",
                    "gpt2_profile", "roofs"]


def run_experiments(with_traces: bool = False) -> dict:
    import tempfile

    import torch
    env, spec = _env()
    files = {}
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

    # Chapters 2-5: additional measurements (each in its own child interpreter)
    extra = {}
    for key in EXTRA_PER_REPEAT:
        extra[key] = _run_experiment_subprocess(key, "/root/experiments_extra.py", files=files,
                                                prefix=f"extra/{key}")
    # GPT-2 cold vs warm start: the cold run gets empty Inductor and Triton caches, the warm run reuses
    # them; regional compilation gets its own empty caches.
    cold = {"TORCHINDUCTOR_CACHE_DIR": tempfile.mkdtemp(prefix="ind_"), "TRITON_CACHE_DIR": tempfile.mkdtemp(prefix="tri_")}
    extra["gpt2_cold"] = _run_experiment_subprocess("gpt2_cold", "/root/experiments_extra.py", cold)
    extra["gpt2_warm"] = _run_experiment_subprocess("gpt2_warm", "/root/experiments_extra.py", cold)
    regional = {"TORCHINDUCTOR_CACHE_DIR": tempfile.mkdtemp(prefix="ind_"), "TRITON_CACHE_DIR": tempfile.mkdtemp(prefix="tri_")}
    extra["gpt2_regional"] = _run_experiment_subprocess("gpt2_regional", "/root/experiments_extra.py", regional)
    if with_traces:  # profiler traces for Perfetto and an Nsight Systems report, first repeat only
        extra["traces"] = _run_experiment_subprocess("traces", "/root/experiments_extra.py", files=files,
                                                     prefix="pytorch_profiler")
        extra["nsys"] = _run_nsys(files)
    res["extra"] = extra
    res["_files"] = files
    hf_cache.commit()
    return res


def run_allreduce() -> dict:
    """Chapter 7: NCCL all-reduce across every GPU in one container."""
    env, _spec = _env()
    files = {}
    r = _run_experiment_subprocess("allreduce", "/root/experiments_extra.py", files=files, prefix="allreduce")
    return {"env": env, "allreduce": r, "_files": files}


# ------------------------------------------------------------------------------------------------
# per-GPU function registration
# ------------------------------------------------------------------------------------------------
# One pair of Modal functions per configured GPU, each pinned to its GPU at decoration time. This
# replaces Function.with_options(gpu=...), which this Modal build no longer provides, and still lets
# every GPU run in parallel from one app.
LAB_FUNCS, EXP_FUNCS = {}, {}
for _gpu in cfg.GPUS:
    _t = _tag(_gpu)
    _m = getattr(cfg, "MODAL_GPU", {}).get(_gpu, _gpu)
    LAB_FUNCS[_gpu] = app.function(name=f"run_chapter02_lab_{_t}", gpu=_m, **COMMON)(run_chapter02_lab)
    EXP_FUNCS[_gpu] = app.function(name=f"run_experiments_{_t}", gpu=_m, **COMMON)(run_experiments)
ALLREDUCE_GPUS = ["A100-80GB:8", "H100!:8"]
ALLREDUCE_FUNCS = {g: app.function(name=f"run_allreduce_{_tag(g).replace(':', 'x')}", gpu=g, **COMMON)(run_allreduce)
                   for g in ALLREDUCE_GPUS}


# ------------------------------------------------------------------------------------------------
# local entrypoint
# ------------------------------------------------------------------------------------------------
@app.local_entrypoint()
def main(gpus: str = "", repeats: int = 1, allreduce: bool = False):
    """Run every GPU in parallel; with --repeats N, run N independent containers per GPU.

    repeats == 1 writes results/<gpu>.json as before. repeats > 1 also writes every repeat to
    results/repeats/<gpu>_r<k>.json, and results/<gpu>.json becomes the median across repeats
    (results/<gpu>_spread.json holds [min, max] for every number)."""
    import aggregate
    import make_tables
    import principles

    RESULTS.mkdir(parents=True, exist_ok=True)
    targets = [g.strip() for g in gpus.split(",") if g.strip()] or list(cfg.GPUS)
    missing = [g for g in targets if g not in LAB_FUNCS]
    if missing:
        raise SystemExit(f"GPUs not in experiment_config.GPUS: {missing}. Add them there first.")
    repeats = max(1, int(repeats))
    jobs = {g: [(LAB_FUNCS[g].spawn(with_traces=(k == 0)), EXP_FUNCS[g].spawn(with_traces=(k == 0)))
                for k in range(repeats)] for g in targets}
    ar_jobs = {g: f.spawn() for g, f in ALLREDUCE_FUNCS.items()} if allreduce else {}

    def _write_files(tag, file_dict):
        for rel, data in (file_dict or {}).items():
            p = RESULTS / "traces" / tag / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
    by_gpu, runs_by_gpu = {}, {}
    if repeats > 1:
        (RESULTS / "repeats").mkdir(exist_ok=True)
    for g, pairs in jobs.items():
        tag = g.lower().replace("-", "_")
        runs = []
        for k, (lab_job, exp_job) in enumerate(pairs, 1):
            try:
                lab, exp = lab_job.get(), exp_job.get()
            except Exception as e:  # noqa: BLE001  a lost container should not lose the other repeats
                print(f"{g} repeat {k}: FAILED ({e!r})")
                continue
            nb = lab.pop("executed_notebook")
            tf = {**lab.pop("_files", {}), **exp.pop("_files", {})}
            if tf:
                tf["DEVICE.txt"] = ((exp.get("env") or {}).get("gpu", "?") + "\n").encode()
            _write_files(tag, tf)
            res = {"requested_gpu": g, "repeat": k, **exp, "chapter02_lab": lab}
            runs.append(res)
            if repeats > 1:
                (RESULTS / "repeats" / f"{tag}_r{k}.json").write_text(json.dumps(res, indent=2))
                (RESULTS / "repeats" / f"chapter02_lab_executed_{tag}_r{k}.ipynb").write_text(nb)
            if k == 1 or not (RESULTS / f"chapter02_lab_executed_{tag}.ipynb").exists():
                (RESULTS / f"chapter02_lab_executed_{tag}.ipynb").write_text(nb)
            print(f"{g} repeat {k}: lab errors={len(lab['errors'])}; experiment errors="
                  f"{[x for x in ('embedder', 'decode_batching') if 'error' in (res.get(x) or {})]}"
                  f"; decode runs with errors={[x for x, v in res['compiled_decode']['runs'].items() if 'error' in v]}")
        if not runs:
            continue
        want = getattr(cfg, "EXPECTED_DEVICE", {}).get(g, "")
        merged, sp, views, mlog = aggregate.merge_repeats(runs, want)
        for m in mlog:
            print(f"{g} {m}")
        runs_by_gpu[g] = views
        (RESULTS / f"{tag}.json").write_text(json.dumps(merged, indent=2))
        if sp:
            (RESULTS / f"{tag}_spread.json").write_text(json.dumps(sp, indent=2))
        by_gpu[g] = merged
    for g, job in ar_jobs.items():
        try:
            r = job.get()
            _write_files(_tag(g).replace(":", "x"), r.pop("_files", {}))
            (RESULTS / f"allreduce_{_tag(g).replace(':', 'x')}.json").write_text(json.dumps(r, indent=2))
            print(f"all-reduce {g}: {'error' if 'error' in r['allreduce'] else 'ok'}")
        except Exception as e:  # noqa: BLE001
            print(f"all-reduce {g}: FAILED ({e!r})")
    rows = principles.evaluate(by_gpu)
    report = principles.markdown_report(rows, by_gpu)
    if any(len(r) > 1 for r in runs_by_gpu.values()):
        counts = principles.count_across_repeats(runs_by_gpu)
        report += principles.markdown_counts(counts)
        for r in rows:
            r["holds_in_repeats"] = counts.get(r["id"], {})
    (RESULTS / "principles_report.json").write_text(json.dumps(rows, indent=2))
    (RESULTS / "principles_report.md").write_text(report)
    print("\n" + report.split("## Evidence")[0])
    T = make_tables.build()
    (RESULTS / "master_tables.json").write_text(json.dumps(T, indent=2))
    (RESULTS / "master_tables.md").write_text(make_tables.to_md(T))
    print(f"Results in {RESULTS}")
