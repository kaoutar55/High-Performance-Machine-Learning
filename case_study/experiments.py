"""GPU measurements for the case study, with no dependency on Modal.

These functions run inside the GPU container. They are kept free of any `modal` import so that
one of them (the compiled-decode experiment) can be launched in a fresh child interpreter: on some
GPUs torch.compile with CUDA graphs plus the static KV cache crashes the process natively
(SIGSEGV), which a try/except cannot catch. Running it in a subprocess turns that crash into a
non-zero exit the runner records as an error, leaving every other measurement intact.

Run one experiment in isolation (used by modal_case_study.run_experiments):
    python experiments.py <name> <out.json>     # names: see EXPERIMENTS at the bottom
The Chapter 4 decode measurements are split into several experiments so that a native crash in
one compile mode does not lose the others.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, str(HERE))
import experiment_config as cfg  # noqa: E402


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
    # Namespaced repo id: newer datasets/huggingface_hub reject the bare "wikitext"
    # name. Salesforce/wikitext is the canonical mirror, same config and split.
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
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
            # tokenize() returns tensors plus some non-tensor metadata in newer
            # sentence-transformers; move only the tensors to the GPU.
            feats = {k: (v.cuda() if hasattr(v, "cuda") else v)
                     for k, v in m.tokenize(chunks[:32]).items()}
            out["tokens_per_chunk"] = int(feats["input_ids"].shape[1])
            with torch.inference_mode():
                m(feats); torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    m(feats); torch.cuda.synchronize()
            out["profile_fp16_batch32"] = _top_ops(prof)
            keys = [e.key for e in prof.key_averages()]
            names = " ".join(keys).lower()
            out["sdpa_backend_seen"] = {"flash": "flash" in names, "mem_efficient": "efficient" in names}
            att = [k for k in keys if any(w in k.lower() for w in
                   ("attention", "fmha", "flash", "sdpa", "cudnn"))]
            out["attention_ops"] = att[:10]
            out["attention_backend"] = (
                "cudnn" if any("cudnn_attention" in k for k in att) else
                "flash" if any("flash_attention" in k for k in att) else
                "mem_efficient" if any("efficient_attention" in k for k in att) else
                "math (unfused)" if not att else "other")
            out["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        del m; torch.cuda.empty_cache()
    return out


def _graph_count():
    """Number of graphs Dynamo has compiled in this process (0 means nothing was compiled)."""
    try:
        from torch._dynamo.utils import counters
        return int(counters["stats"]["unique_graphs"])
    except Exception:  # noqa: BLE001
        return None


def _autocompile_kw():
    """generate() kwargs that switch off transformers' automatic compilation for a static cache.

    Recent transformers compile the forward pass inside generate() whenever the cache is static.
    To measure eager with a static cache we must turn that off; record whether the switch exists."""
    try:
        from transformers import GenerationConfig
        if hasattr(GenerationConfig(), "disable_compile"):
            return {"disable_compile": True}
    except Exception:  # noqa: BLE001
        pass
    return None


def _decode_setup(on_progress):
    import torch
    tok, model = _generator()
    inputs = _prompt(tok)
    out = {"prompt_tokens": int(inputs["input_ids"].shape[1]), "new_tokens": cfg.NEW_TOKENS}

    def flush():
        if on_progress is not None:
            on_progress(out)

    def gen(m, n, **kw):
        return lambda: m.generate(**inputs, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                                  pad_token_id=tok.eos_token_id, **kw)

    def measure(label, m, **kw):
        import time
        g0 = _graph_count()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        gen(m, cfg.NEW_TOKENS, **kw)()          # first call: includes any compilation
        torch.cuda.synchronize(); first = time.perf_counter() - t0
        t1 = _cuda_time(gen(m, 1, **kw))
        tn = _cuda_time(gen(m, cfg.NEW_TOKENS, **kw))
        g1 = _graph_count()
        out[label] = {"t_first_token_s": round(t1, 4), "t_all_tokens_s": round(tn, 4),
                      "decode_ms_per_token": round((tn - t1) / (cfg.NEW_TOKENS - 1) * 1000, 2),
                      "first_call_s": round(first, 2),
                      "dynamo_graphs_compiled": (None if None in (g0, g1) else g1 - g0)}
        flush()

    return torch, model, out, flush, measure


def exp_decode_eager(on_progress=None) -> dict:
    """Chapter 4 baseline: eager decode with the dynamic cache, eager with a static cache (automatic
    compilation off), and the static cache with transformers' automatic compilation left on."""
    torch, model, out, flush, measure = _decode_setup(on_progress)
    off = _autocompile_kw()
    out["autocompile_switch_available"] = off is not None
    with torch.inference_mode():
        measure("eager_dynamic_cache", model)
        if off is not None:
            measure("eager_static_cache", model, cache_implementation="static", **off)
        torch._dynamo.reset()
        measure("hf_autocompile_static_cache", model, cache_implementation="static")
    flush()
    return out


def exp_decode_wrong_compile(on_progress=None) -> dict:
    """Chapter 4: compile the module, then call .generate(). generate() runs outside the compiled
    function, so decode should be essentially unchanged."""
    torch, model, out, flush, measure = _decode_setup(on_progress)
    with torch.inference_mode():
        measure("wrong_compile_module_then_generate", torch.compile(model, mode="reduce-overhead"))
    flush()
    return out


def _right_compile(mode, on_progress):
    torch, model, out, flush, measure = _decode_setup(on_progress)
    off = _autocompile_kw() or {}
    out["mode"] = mode
    with torch.inference_mode():
        model.forward = torch.compile(model.forward, mode=mode)
        measure("right_compile_forward_static_cache", model, cache_implementation="static", **off)
    flush()
    return out


def exp_decode_compile_reduce_overhead(on_progress=None) -> dict:
    """Chapter 4: compile model.forward with CUDA graphs (mode="reduce-overhead") + static cache."""
    return _right_compile("reduce-overhead", on_progress)


def exp_decode_compile_default(on_progress=None) -> dict:
    """Chapter 4 fallback: compile model.forward without CUDA graphs (mode="default") + static cache."""
    return _right_compile("default", on_progress)


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


EXPERIMENTS = {"embedder": exp_embedder, "decode_batching": exp_decode_batching,
               "decode_eager": exp_decode_eager, "decode_wrong_compile": exp_decode_wrong_compile,
               "decode_compile_reduce_overhead": exp_decode_compile_reduce_overhead,
               "decode_compile_default": exp_decode_compile_default}
DECODE_EXPERIMENTS = [k for k in EXPERIMENTS if k.startswith("decode_") and k != "decode_batching"]


if __name__ == "__main__":
    # Run one experiment in this (isolated) interpreter and write its result as JSON. For
    # compiled_decode, flush partial results after each measurement so a native crash in a later
    # step still leaves the earlier numbers on disk.
    import json

    name, out_path = sys.argv[1], sys.argv[2]
    fn = EXPERIMENTS[name]
    if name in DECODE_EXPERIMENTS:
        def _save(partial):
            json.dump(partial, open(out_path, "w"))
        res = fn(on_progress=_save)
    else:
        res = fn()
    json.dump(res, open(out_path, "w"))
