"""Re-measure the book's RAG case study on an NVIDIA T4 with Modal.

What it runs (all on one T4 each, in parallel):
  1. chapter02/chapter02_lab.ipynb, executed top to bottom (Gemma-2B-IT FP16 + BGE-M3 + FAISS).
     Produces Table 2.6 (latency distribution), Table 2.7 / Figure 2.9 (stage table),
     Table 2.4 (profiler), Table 2.8 (roofline placement). Saved as an executed notebook.
  2. Extra measurements the chapters need that no notebook produces yet:
     - Chapter 3: BGE-M3 throughput on 512-token chunks (FP32 vs FP16, batch sweep), model shape,
       operator profile and which SDPA attention backend runs on a T4.
     - Chapter 4: Gemma decode eager vs torch.compile, including the wrong pattern
       (torch.compile(model).generate) and the right one (compile model.forward + static cache).
     - Chapter 5: Gemma prefill and per-token decode time vs batch size, achieved bandwidth.

Prerequisites (once):  pip install modal && modal setup
                       modal secret create huggingface HF_TOKEN=<read token>   (accept the Gemma license first)
Run from the repository root:
    modal run case_study/modal_case_study.py              # T4 (the Colab free-tier reference)
    modal run case_study/modal_case_study.py --gpu A100-80GB   # same measurements on an A100
Outputs (in case_study/results/):  case_study_results_<gpu>.json, chapter02_lab_executed_<gpu>.ipynb
"""
import json
import os
import pathlib

import modal

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

hf_cache = modal.Volume.from_name("hpml-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "accelerate", "sentence-transformers", "faiss-cpu", "datasets",
        "numpy", "pandas", "matplotlib", "nbclient", "nbformat", "ipykernel",
    )
    .run_commands("python -m ipykernel install --name python3 --display-name 'Python 3'")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "MPLBACKEND": "Agg"})
    .add_local_dir(str(REPO / "chapter02"), remote_path="/repo/chapter02")
)

app = modal.App("hpml-case-study", image=image)
COMMON = dict(gpu="T4", timeout=3 * 3600, volumes={"/cache": hf_cache},
              secrets=[modal.Secret.from_name("huggingface")])

GEMMA = "google/gemma-2b-it"
BGE = "BAAI/bge-m3"


def _env():
    import platform
    import subprocess

    import sentence_transformers
    import torch
    import transformers
    return {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "driver": subprocess.getoutput("nvidia-smi --query-gpu=driver_version --format=csv,noheader").strip(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__, "sentence_transformers": sentence_transformers.__version__,
        "python": platform.python_version(),
    }


# ----------------------------------------------------------------------------------------------
# 1. Execute the Chapter 2 lab notebook
# ----------------------------------------------------------------------------------------------
@app.function(**COMMON)
def run_chapter02_lab() -> dict:
    import nbformat
    from nbclient import NotebookClient

    os.chdir("/repo/chapter02")
    nb = nbformat.read("chapter02_lab.ipynb", as_version=4)
    # The image already has the libraries; skip in-notebook pip installs (they would also
    # try to change the torch build).
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.source = "\n".join(("# [skipped on Modal] " + ln) if ln.lstrip().startswith(("!", "%pip"))
                                    else ln for ln in cell.source.split("\n"))
    client = NotebookClient(nb, timeout=3600, kernel_name="python3", allow_errors=True,
                            resources={"metadata": {"path": "/repo/chapter02"}})
    client.execute()
    errors = []
    for i, cell in enumerate(nb.cells):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                errors.append({"cell": i, "ename": out.get("ename"), "evalue": out.get("evalue", "")[:500]})
    hf_cache.commit()
    return {"executed_notebook": nbformat.writes(nb), "errors": errors, "env": _env()}


# ----------------------------------------------------------------------------------------------
# 2. Extra measurements (Chapters 3, 4, 5)
# ----------------------------------------------------------------------------------------------
def _cuda_time(fn, warmup=2, iters=5):
    """Median wall time (s) of fn() with CUDA synchronization around each call."""
    import statistics
    import time

    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def _wikitext_text(min_chars=200, max_chars=4000):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    return [t.strip() for t in ds["text"] if min_chars < len(t) < max_chars]


def _top_ops(prof, n=15):
    rows = []
    for e in sorted(prof.key_averages(), key=lambda e: getattr(e, "self_device_time_total",
                                                               getattr(e, "self_cuda_time_total", 0)),
                    reverse=True)[:n]:
        dev = getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0))
        rows.append({"op": e.key, "self_cuda_ms": round(dev / 1000, 3), "calls": e.count})
    return rows


def ch3_bge_m3() -> dict:
    import torch
    from sentence_transformers import SentenceTransformer
    from torch.profiler import ProfilerActivity, profile

    out = {}
    probe = SentenceTransformer(BGE, device="cuda")
    cfg = probe[0].auto_model.config
    out["model"] = {"id": BGE, "layers": cfg.num_hidden_layers, "heads": cfg.num_attention_heads,
                    "hidden": cfg.hidden_size, "params_m": round(sum(p.numel() for p in probe.parameters()) / 1e6, 1),
                    "max_position_embeddings": cfg.max_position_embeddings}
    tok = probe.tokenizer
    # Exactly-512-token chunks cut from wikitext (decoded back to text, re-tokenized by the model).
    ids = tok("\n".join(_wikitext_text()[:3000]), add_special_tokens=False)["input_ids"]
    chunks = [tok.decode(ids[i:i + 500]) for i in range(0, 500 * 256, 500)]
    del probe; torch.cuda.empty_cache()

    out["throughput"] = []
    for dtype_name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        m = SentenceTransformer(BGE, device="cuda", model_kwargs={"torch_dtype": dtype})
        m.max_seq_length = 512
        for bs in (1, 8, 32, 64):
            n = 64 if bs == 1 else 256
            def run():
                m.encode(chunks[:n], batch_size=bs, convert_to_numpy=True, normalize_embeddings=True)
            t = _cuda_time(run, warmup=1, iters=3)
            out["throughput"].append({"dtype": dtype_name, "batch": bs, "chunks": n, "seconds": round(t, 3),
                                      "chunks_per_s": round(n / t, 1)})
        if dtype_name == "fp16":
            feats = m.tokenize(chunks[:32]); feats = {k: v.cuda() for k, v in feats.items()}
            out["tokens_per_chunk"] = int(feats["input_ids"].shape[1])
            with torch.inference_mode():
                m(feats); torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    m(feats); torch.cuda.synchronize()
            out["profile_fp16_batch32"] = _top_ops(prof)
            names = " ".join(e.key for e in prof.key_averages()).lower()
            out["sdpa_backend_seen"] = {"flash": "flash" in names, "mem_efficient": "efficient" in names,
                                        "math_softmax": "softmax" in names}
            out["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        del m; torch.cuda.empty_cache()
    return out


def _gemma(attn="sdpa"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(GEMMA)
    model = AutoModelForCausalLM.from_pretrained(GEMMA, torch_dtype=torch.float16, device_map="cuda",
                                                 attn_implementation=attn).eval()
    return tok, model


def _prompt(tok, n_tokens=512, batch=1):
    text = "Context:\n" + "\n\n".join(_wikitext_text()[:40]) + "\n\nQuestion: What is described above?\nAnswer:"
    ids = tok(text, return_tensors="pt", truncation=True, max_length=n_tokens)
    return {k: v.repeat(batch, 1).cuda() for k, v in ids.items()}


def ch4_compiled_decode() -> dict:
    import time

    import torch
    tok, model = _gemma()
    inputs = _prompt(tok, 512)
    out = {"prompt_tokens": int(inputs["input_ids"].shape[1]), "new_tokens": 20}

    def gen(m, n, **kw):
        return lambda: m.generate(**inputs, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                                  pad_token_id=tok.eos_token_id, **kw)

    def measure(label, m, **kw):
        t1 = _cuda_time(gen(m, 1, **kw), warmup=2, iters=5)
        t20 = _cuda_time(gen(m, 20, **kw), warmup=2, iters=5)
        out[label] = {"t_1_token_s": round(t1, 4), "t_20_tokens_s": round(t20, 4),
                      "decode_ms_per_token": round((t20 - t1) / 19 * 1000, 2)}

    with torch.inference_mode():
        measure("eager_dynamic_cache", model)
        measure("eager_static_cache", model, cache_implementation="static")

        # Wrong pattern used in the chapter: compile the module, call .generate on the wrapper.
        torch._dynamo.reset()
        wrapped = torch.compile(model, mode="reduce-overhead")
        try:
            measure("wrong_compile_module_then_generate", wrapped)
        except Exception as e:  # noqa: BLE001
            out["wrong_compile_module_then_generate"] = {"error": repr(e)[:400]}

        # Right pattern: compile forward, static KV cache so shapes stay fixed for CUDA graphs.
        torch._dynamo.reset()
        orig_forward = model.forward
        for fullgraph in (True, False):
            try:
                model.forward = torch.compile(orig_forward, mode="reduce-overhead", fullgraph=fullgraph)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                gen(model, 20, cache_implementation="static")()
                torch.cuda.synchronize()
                out["right_compile_first_call_s"] = round(time.perf_counter() - t0, 1)
                measure("right_compile_forward_static_cache", model, cache_implementation="static")
                out["right_compile_fullgraph"] = fullgraph
                break
            except Exception as e:  # noqa: BLE001
                out[f"right_compile_error_fullgraph_{fullgraph}"] = repr(e)[:400]
                torch._dynamo.reset(); model.forward = orig_forward
        model.forward = orig_forward
    return out


def ch5_gemma_batching() -> dict:
    import torch
    tok, model = _gemma()
    n_params = sum(p.numel() for p in model.parameters())
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    out = {"params_b": round(n_params / 1e9, 3), "weight_gb": round(weight_bytes / 1e9, 2), "rows": []}
    with torch.inference_mode():
        for bs in (1, 4, 16, 64):
            try:
                inputs = _prompt(tok, 512, batch=bs)
                f = lambda n: (lambda: model.generate(**inputs, max_new_tokens=n, min_new_tokens=n,
                                                      do_sample=False, pad_token_id=tok.eos_token_id))
                t1 = _cuda_time(f(1), warmup=1, iters=3)
                t20 = _cuda_time(f(20), warmup=1, iters=3)
                per_tok = (t20 - t1) / 19
                prompt_len = int(inputs["input_ids"].shape[1])
                out["rows"].append({
                    "batch": bs, "prompt_tokens": prompt_len,
                    "prefill_s": round(t1, 4),
                    "prefill_tflops": round(2 * n_params * prompt_len * bs / t1 / 1e12, 2),
                    "decode_ms_per_step": round(per_tok * 1000, 2),
                    "decode_tokens_per_s": round(bs / per_tok, 1),
                    "weight_bw_gbs_at_this_rate": round(weight_bytes / per_tok / 1e9, 1),
                })
            except torch.cuda.OutOfMemoryError:
                out["rows"].append({"batch": bs, "error": "OOM"}); torch.cuda.empty_cache()
    return out


@app.function(**COMMON)
def run_extras() -> dict:
    res = {"env": _env()}
    for name, fn in (("chapter03_bge_m3", ch3_bge_m3), ("chapter04_compiled_decode", ch4_compiled_decode),
                     ("chapter05_gemma_batching", ch5_gemma_batching)):
        try:
            res[name] = fn()
        except Exception as e:  # noqa: BLE001  keep going; report the failure
            import traceback
            res[name] = {"error": repr(e)[:500], "traceback": traceback.format_exc()[-2000:]}
        import torch
        torch.cuda.empty_cache()
    hf_cache.commit()
    return res


@app.local_entrypoint()
def main(gpu: str = "T4"):
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = gpu.lower().replace("-", "_")
    lab = run_chapter02_lab.with_options(gpu=gpu).spawn()
    extras = run_extras.with_options(gpu=gpu).spawn()
    lab_res, extras_res = lab.get(), extras.get()
    (RESULTS / f"chapter02_lab_executed_{tag}.ipynb").write_text(lab_res.pop("executed_notebook"))
    summary = {"requested_gpu": gpu, "chapter02_lab": lab_res, **extras_res}
    (RESULTS / f"case_study_results_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: (v if k in ("env",) else "ok" if "error" not in str(v)[:200] else "check")
                      for k, v in summary.items()}, indent=2))
    print(f"\nWrote {RESULTS / f'case_study_results_{tag}.json'} and the executed Chapter 2 notebook.")
    print("Send both files back to Claude.")
