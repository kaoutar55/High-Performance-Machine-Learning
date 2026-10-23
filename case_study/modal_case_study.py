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
    .add_local_dir(str(REPO / "chapter02"), remote_path="/repo/chapter02")
)

app = modal.App("hpml-case-study", image=image)
COMMON = dict(gpu="T4", timeout=3 * 3600, volumes={"/cache": hf_cache},
              secrets=[modal.Secret.from_name("huggingface")])


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
    }, spec


def _dtype():
    import torch
    return getattr(torch, cfg.DTYPE)


def _cuda_time(fn, warmup=None, iters=None):
    """Median wall time (s) of fn(), synchronizing the GPU around each call."""
    import statistics
    import time

    import torch
    warmup = cfg.TIMING["warmup"] if warmup is None else warmup
    iters = cfg.TIMING["iters"] if iters is None else iters
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def _wikitext(min_chars=200, max_chars=4000):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    return [t.strip() for t in ds["text"] if min_chars < len(t) < max_chars]


def _top_ops(prof, n=15):
    def dev(e):
        return getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
    return [{"op": e.key, "self_cuda_ms": round(dev(e) / 1000, 3), "calls": e.count}
            for e in sorted(prof.key_averages(), key=dev, reverse=True)[:n]]


def _generator():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.MODELS["generator"])
    model = AutoModelForCausalLM.from_pretrained(cfg.MODELS["generator"], torch_dtype=_dtype(),
                                                 device_map="cuda", attn_implementation="sdpa").eval()
    return tok, model


def _prompt(tok, batch=1):
    text = ("Context:\n" + "\n\n".join(_wikitext()[:40]) + "\n\nQuestion: What is described above?\nAnswer:")
    ids = tok(text, return_tensors="pt", truncation=True, max_length=cfg.PROMPT_TOKENS)
    return {k: v.repeat(batch, 1).cuda() for k, v in ids.items()}


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


@app.function(**COMMON)
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
def exp_embedder() -> dict:
    """Chapter 3: embedder shape, throughput vs precision and batch, profile, attention kernel."""
    import torch
    from sentence_transformers import SentenceTransformer
    from torch.profiler import ProfilerActivity, profile

    mid = cfg.MODELS["embedder"]
    probe = SentenceTransformer(mid, device="cuda")
    c = probe[0].auto_model.config
    out = {"model": {"id": mid, "layers": c.num_hidden_layers, "heads": c.num_attention_heads,
                     "hidden": c.hidden_size, "params_m": round(sum(p.numel() for p in probe.parameters()) / 1e6, 1)}}
    tok = probe.tokenizer
    ids = tok("\n".join(_wikitext()[:3000]), add_special_tokens=False)["input_ids"]
    step = cfg.EMBED_CHUNK_TOKENS - 12  # leave room for special tokens
    chunks = [tok.decode(ids[i:i + step]) for i in range(0, step * cfg.EMBED_CHUNKS, step)]
    del probe; torch.cuda.empty_cache()

    out["throughput"] = []
    for name, dt in (("fp32", torch.float32), ("fp16", _dtype())):
        m = SentenceTransformer(mid, device="cuda", model_kwargs={"torch_dtype": dt})
        m.max_seq_length = cfg.EMBED_CHUNK_TOKENS
        for bs in cfg.EMBED_BATCHES:
            n = 64 if bs == 1 else cfg.EMBED_CHUNKS
            t = _cuda_time(lambda: m.encode(chunks[:n], batch_size=bs, convert_to_numpy=True,
                                            normalize_embeddings=True), warmup=1, iters=3)
            out["throughput"].append({"dtype": name, "batch": bs, "chunks": n, "seconds": round(t, 3),
                                      "chunks_per_s": round(n / t, 1)})
        if name == "fp16":
            feats = {k: v.cuda() for k, v in m.tokenize(chunks[:32]).items()}
            out["tokens_per_chunk"] = int(feats["input_ids"].shape[1])
            with torch.inference_mode():
                m(feats); torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    m(feats); torch.cuda.synchronize()
            out["profile_fp16_batch32"] = _top_ops(prof)
            names = " ".join(e.key for e in prof.key_averages()).lower()
            out["sdpa_backend_seen"] = {"flash": "flash" in names, "mem_efficient": "efficient" in names}
            out["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        del m; torch.cuda.empty_cache()
    return out


def exp_compiled_decode() -> dict:
    """Chapter 4: eager vs compiled decode, including the pattern that silently compiles nothing."""
    import time

    import torch
    tok, model = _generator()
    inputs = _prompt(tok)
    out = {"prompt_tokens": int(inputs["input_ids"].shape[1]), "new_tokens": cfg.NEW_TOKENS}

    def gen(m, n, **kw):
        return lambda: m.generate(**inputs, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                                  pad_token_id=tok.eos_token_id, **kw)

    def measure(label, m, **kw):
        t1 = _cuda_time(gen(m, 1, **kw))
        tn = _cuda_time(gen(m, cfg.NEW_TOKENS, **kw))
        out[label] = {"t_first_token_s": round(t1, 4), "t_all_tokens_s": round(tn, 4),
                      "decode_ms_per_token": round((tn - t1) / (cfg.NEW_TOKENS - 1) * 1000, 2)}

    with torch.inference_mode():
        measure("eager_dynamic_cache", model)
        measure("eager_static_cache", model, cache_implementation="static")
        torch._dynamo.reset()
        try:
            measure("wrong_compile_module_then_generate", torch.compile(model, mode="reduce-overhead"))
        except Exception as e:  # noqa: BLE001
            out["wrong_compile_module_then_generate"] = {"error": repr(e)[:400]}
        torch._dynamo.reset()
        orig = model.forward
        for fullgraph in (True, False):
            try:
                model.forward = torch.compile(orig, mode="reduce-overhead", fullgraph=fullgraph)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                gen(model, cfg.NEW_TOKENS, cache_implementation="static")()
                torch.cuda.synchronize()
                out["right_compile_first_call_s"] = round(time.perf_counter() - t0, 1)
                measure("right_compile_forward_static_cache", model, cache_implementation="static")
                out["right_compile_fullgraph"] = fullgraph
                break
            except Exception as e:  # noqa: BLE001
                out[f"right_compile_error_fullgraph_{fullgraph}"] = repr(e)[:400]
                torch._dynamo.reset(); model.forward = orig
        model.forward = orig
    return out


def exp_decode_batching() -> dict:
    """Chapter 5: prefill and per-token decode against batch size, with achieved compute and bandwidth."""
    import torch
    tok, model = _generator()
    params = sum(p.numel() for p in model.parameters())
    wbytes = sum(p.numel() * p.element_size() for p in model.parameters())
    out = {"params_b": round(params / 1e9, 3), "weight_gb": round(wbytes / 1e9, 3), "rows": []}
    with torch.inference_mode():
        for bs in cfg.DECODE_BATCHES:
            try:
                inputs = _prompt(tok, batch=bs)
                def f(n):
                    return lambda: model.generate(**inputs, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                                                  pad_token_id=tok.eos_token_id)
                t1 = _cuda_time(f(1), warmup=1, iters=3)
                tn = _cuda_time(f(cfg.NEW_TOKENS), warmup=1, iters=3)
                step = (tn - t1) / (cfg.NEW_TOKENS - 1)
                plen = int(inputs["input_ids"].shape[1])
                out["rows"].append({"batch": bs, "prompt_tokens": plen, "prefill_s": round(t1, 4),
                                    "prefill_tflops": round(2 * params * plen * bs / t1 / 1e12, 2),
                                    "decode_ms_per_step": round(step * 1000, 2),
                                    "decode_tokens_per_s": round(bs / step, 1),
                                    "weight_read_gbs": round(wbytes / step / 1e9, 1)})
            except torch.cuda.OutOfMemoryError:
                out["rows"].append({"batch": bs, "error": "out of memory"}); torch.cuda.empty_cache()
    return out


@app.function(**COMMON)
def run_experiments() -> dict:
    import traceback

    import torch
    env, spec = _env()
    res = {"env": env, "spec": spec, "config": {k: getattr(cfg, k) for k in
           ("MODELS", "DTYPE", "PROMPT_TOKENS", "NEW_TOKENS", "DECODE_BATCHES", "EMBED_BATCHES", "TIMING")}}
    for key, fn in (("embedder", exp_embedder), ("compiled_decode", exp_compiled_decode),
                    ("decode_batching", exp_decode_batching)):
        try:
            res[key] = fn()
        except Exception as e:  # noqa: BLE001  keep going; the failure is reported
            res[key] = {"error": repr(e)[:500], "traceback": traceback.format_exc()[-2000:]}
        torch.cuda.empty_cache()
    hf_cache.commit()
    return res


# ------------------------------------------------------------------------------------------------
# local entrypoint
# ------------------------------------------------------------------------------------------------
@app.local_entrypoint()
def main(gpus: str = ""):
    import principles

    RESULTS.mkdir(parents=True, exist_ok=True)
    targets = [g.strip() for g in gpus.split(",") if g.strip()] or list(cfg.GPUS)
    jobs = {g: (run_chapter02_lab.with_options(gpu=g).spawn(), run_experiments.with_options(gpu=g).spawn())
            for g in targets}
    by_gpu = {}
    for g, (lab_job, exp_job) in jobs.items():
        tag = g.lower().replace("-", "_")
        lab, exp = lab_job.get(), exp_job.get()
        (RESULTS / f"chapter02_lab_executed_{tag}.ipynb").write_text(lab.pop("executed_notebook"))
        res = {"requested_gpu": g, **exp, "chapter02_lab": lab}
        (RESULTS / f"{tag}.json").write_text(json.dumps(res, indent=2))
        by_gpu[g] = res
        print(f"{g}: lab errors={len(lab['errors'])}; experiment errors="
              f"{[k for k in ('embedder', 'compiled_decode', 'decode_batching') if 'error' in (res.get(k) or {})]}")
    rows = principles.evaluate(by_gpu)
    (RESULTS / "principles_report.json").write_text(json.dumps(rows, indent=2))
    (RESULTS / "principles_report.md").write_text(principles.markdown_report(rows, by_gpu))
    print("\n" + principles.markdown_report(rows, by_gpu).split("## Evidence")[0])
    print(f"Results in {RESULTS}")
