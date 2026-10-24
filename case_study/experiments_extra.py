"""Additional measurements for Chapters 2-5 and 7, plus profiler and Nsight Systems traces.

Like experiments.py, nothing here imports modal, and every experiment runs in its own child
interpreter (python experiments_extra.py <name> <out.json>). Files an experiment produces (Chrome
traces for Perfetto, profiler tables) go to the directory named by the EXP_OUT_DIR environment
variable; the runner collects them.

Chapter map
  Ch 2  warmup            first call vs later calls of one cuBLAS op (Figure 2.4)
        traces            PyTorch profiler Chrome traces (open in https://ui.perfetto.dev):
                          eager prefill + 5 NVTX-labelled decode steps, compiled decode, embedder
        nsys_target       the program Nsight Systems profiles (prefill + 5 decode steps, NVTX)
  Ch 3  python_loop       Python loop vs sum() vs NumPy vs torch.sum (Table 3.1)
        autograd          Gemma-2B prefill with grad, no_grad, inference_mode: time + memory (Table 3.5)
        dataloader        DataLoader workers / pin_memory / prefetch sweep feeding BGE-M3 (Table 3.7)
        (prefill operator table for Table 3.10 comes from the "traces" experiment)
  Ch 4  compile_stages    eager vs compiled BGE-M3 encoder and Gemma prefill
        gpt2_explain      torch._dynamo.explain on the chapter's GPT-2 with and without graph breaks
        gpt2_cold         first call / steady call, full-model compile, empty inductor cache
        gpt2_warm         same, with the cache the cold run left behind (cold vs warm start)
        gpt2_regional     first call with regional compilation (each block compiled), empty cache
        gpt2_profile      eager vs compiled key_averages tables and Chrome traces
  Ch 5  roofs             measured GEMM TFLOP/s per precision, copy bandwidth, launch overhead
  Ch 7  allreduce         NCCL all-reduce bus bandwidth across all GPUs in the container
"""
import gzip
import json
import os
import shutil
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import experiment_config as cfg  # noqa: E402
from experiments import _cuda_time, _dtype, _generator, _prompt, _top_ops, _wikitext  # noqa: E402

OUT = os.environ.get("EXP_OUT_DIR", "/tmp/exp_out")
GPT2 = "openai-community/gpt2"
GPT2_SEQ = 512


def _out(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def _save_trace(prof, name):
    """Chrome trace, gzipped (Perfetto opens .json.gz directly)."""
    raw = _out(name + ".json")
    prof.export_chrome_trace(raw)
    with open(raw, "rb") as f, gzip.open(raw + ".gz", "wb") as g:
        shutil.copyfileobj(f, g)
    os.remove(raw)
    return name + ".json.gz"


def _table(prof, name, rows=15):
    for key in ("self_cuda_time_total", "self_device_time_total"):
        try:
            txt = prof.key_averages().table(sort_by=key, row_limit=rows)
            break
        except Exception:  # noqa: BLE001
            txt = None
    if txt:
        with open(_out(name + ".txt"), "w") as f:
            f.write(txt)
    return name + ".txt"


def _sync():
    import torch
    torch.cuda.synchronize()


def _manual_decode(model, inputs, steps, nvtx=False, record=False):
    """Prefill, then `steps` single-token decode steps with the KV cache (as in the Chapter 2 lab)."""
    import torch
    from torch.profiler import record_function
    rng = torch.cuda.nvtx
    with torch.inference_mode():
        if nvtx: rng.range_push("prefill")
        if record:
            with record_function("prefill"):
                out = model(**inputs, use_cache=True)
        else:
            out = model(**inputs, use_cache=True)
        if nvtx: rng.range_pop()
        past, tok = out.past_key_values, out.logits[:, -1:].argmax(dim=-1)
        for i in range(steps):
            if nvtx: rng.range_push(f"decode_step_{i}")
            if record:
                with record_function(f"decode_step_{i}"):
                    out = model(input_ids=tok, past_key_values=past, use_cache=True)
            else:
                out = model(input_ids=tok, past_key_values=past, use_cache=True)
            past, tok = out.past_key_values, out.logits[:, -1:].argmax(dim=-1)
            if nvtx: rng.range_pop()
    return tok


# ------------------------------------------------------------------------------------------------
# Chapter 2
# ------------------------------------------------------------------------------------------------
def exp_warmup():
    """Six consecutive calls of one batched matmul in a fresh process: the first pays for CUDA
    context work, cuBLAS handle creation and kernel selection."""
    import torch
    x = torch.randn(32, 512, 512, device="cuda")
    _sync()
    times = []
    for _ in range(6):
        t0 = time.perf_counter(); torch.bmm(x, x); _sync(); times.append((time.perf_counter() - t0) * 1e6)
    return {"op": "torch.bmm(32x512x512, fp32)", "call_us": [round(t, 1) for t in times],
            "first_over_steady": round(times[0] / statistics.median(times[2:]), 1)}


def exp_traces():
    """PyTorch profiler traces for Perfetto plus the matching key_averages tables."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    act = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    tok, model = _generator()
    inputs = _prompt(tok)
    files = []
    for _ in range(2):
        _manual_decode(model, inputs, 5)
    _sync()
    with profile(activities=act, record_shapes=True) as prof:
        _manual_decode(model, inputs, 5, nvtx=True, record=True); _sync()
    files += [_save_trace(prof, "gemma_prefill_decode5_eager"), _table(prof, "gemma_prefill_decode5_eager")]
    out = {"decode_top_ops": _top_ops(prof)}

    # prefill alone (Table 3.10 operator shares)
    with torch.inference_mode():
        model(**inputs, use_cache=False); _sync()
        with profile(activities=act, record_shapes=True) as prof:
            model(**inputs, use_cache=False); _sync()
    files += [_save_trace(prof, "gemma_prefill_eager"), _table(prof, "gemma_prefill_eager")]
    out["prefill_top_ops"] = _top_ops(prof, 20)

    # compiled decode through transformers' static-cache compilation
    def gen(n):
        return model.generate(**inputs, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                              pad_token_id=tok.eos_token_id, cache_implementation="static")
    with torch.inference_mode():
        gen(6); gen(6); _sync()
        with profile(activities=act) as prof:
            gen(6); _sync()
    files += [_save_trace(prof, "gemma_generate6_static_autocompile"),
              _table(prof, "gemma_generate6_static_autocompile")]
    out["compiled_decode_top_ops"] = _top_ops(prof)
    del model; torch.cuda.empty_cache()

    # embedder, FP16, batch 32
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(cfg.MODELS["embedder"], device="cuda", model_kwargs={"torch_dtype": _dtype()})
    m.max_seq_length = cfg.EMBED_CHUNK_TOKENS
    chunks = _chunks(m.tokenizer, 32)
    feats = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in m.tokenize(chunks).items()}
    with torch.inference_mode():
        m(feats); _sync()
        with profile(activities=act, record_shapes=True) as prof:
            m(feats); _sync()
    files += [_save_trace(prof, "bge_m3_fp16_batch32"), _table(prof, "bge_m3_fp16_batch32")]
    out["files"] = files
    return out


def exp_nsys_target():
    """Program profiled by Nsight Systems: the capture range is prefill + 5 NVTX-labelled steps."""
    import torch
    tok, model = _generator()
    inputs = _prompt(tok)
    for _ in range(3):
        _manual_decode(model, inputs, 5)
    _sync()
    torch.cuda.cudart().cudaProfilerStart()
    _manual_decode(model, inputs, 5, nvtx=True)
    _sync()
    torch.cuda.cudart().cudaProfilerStop()
    return {"captured": "prefill + decode_step_0..4 (eager, dynamic cache)"}


# ------------------------------------------------------------------------------------------------
# Chapter 3
# ------------------------------------------------------------------------------------------------
def exp_python_loop():
    """Summing 10 million floats: interpreter loop vs C loop vs vectorized libraries."""
    import numpy as np
    import torch
    n = 10_000_000
    xs = [float(i % 1000) for i in range(n)]
    arr = np.asarray(xs)
    tc = torch.tensor(arr)
    tg = tc.cuda()

    def timed(fn, reps=3):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return round(statistics.median(ts) * 1000, 2)

    def loop():
        s = 0.0
        for v in xs:
            s += v
        return s
    res = {"n": n, "python": __import__("platform").python_version(),
           "ms": {"python_for_loop": timed(loop), "builtin_sum": timed(lambda: sum(xs)),
                  "numpy_sum": timed(lambda: arr.sum())}}
    threads = torch.get_num_threads()
    res["ms"][f"torch_sum_cpu_{threads}_threads"] = timed(lambda: tc.sum())
    torch.set_num_threads(1)
    res["ms"]["torch_sum_cpu_1_thread"] = timed(lambda: tc.sum())
    torch.set_num_threads(threads)
    tg.sum(); _sync()
    res["ms"]["torch_sum_gpu"] = timed(lambda: (tg.sum(), _sync()))
    return res


def exp_autograd():
    """Gemma-2B prefill (512 tokens): autograd recording vs no_grad vs inference_mode."""
    import torch
    tok, model = _generator()
    inputs = _prompt(tok)
    out = {}
    for label, ctx in (("grad_enabled", torch.enable_grad), ("no_grad", torch.no_grad),
                       ("inference_mode", torch.inference_mode)):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()

        def run():
            with ctx():
                y = model(**inputs, use_cache=False).logits
                return y
        t = _cuda_time(run, warmup=2, iters=10)
        torch.cuda.reset_peak_memory_stats()
        with ctx():
            y = model(**inputs, use_cache=False).logits
            _sync()
            peak = torch.cuda.max_memory_allocated()
            out[label] = {"ms": round(t * 1000, 2), "peak_extra_gb": round((peak - base) / 1e9, 3),
                          "output_requires_grad": bool(y.requires_grad)}
        del y
    return out


def _chunks(tok, n):
    ids = tok("\n".join(_wikitext()[:3000]), add_special_tokens=False)["input_ids"]
    step = cfg.EMBED_CHUNK_TOKENS - 12
    return [tok.decode(ids[i:i + step]) for i in range(0, step * n, step)]


def exp_dataloader():
    """Feed BGE-M3 (FP16, batch 32) from a DataLoader that tokenizes raw text in __getitem__."""
    import torch
    from sentence_transformers import SentenceTransformer
    from torch.utils.data import DataLoader, Dataset

    m = SentenceTransformer(cfg.MODELS["embedder"], device="cuda", model_kwargs={"torch_dtype": _dtype()})
    tok = m.tokenizer
    texts = _chunks(tok, 1024)
    enc = m[0].auto_model

    class Texts(Dataset):
        def __len__(self):
            return len(texts)

        def __getitem__(self, i):
            t = tok(texts[i], padding="max_length", truncation=True, max_length=cfg.EMBED_CHUNK_TOKENS,
                    return_tensors="pt")
            return {k: v[0] for k, v in t.items()}

    settings = [(0, False, None), (2, True, 2), (4, False, 2), (4, True, 2), (8, True, 2), (8, True, 4)]
    rows = []
    with torch.inference_mode():
        for workers, pin, prefetch in settings:
            kw = dict(batch_size=32, num_workers=workers, pin_memory=pin)
            if workers:
                kw["prefetch_factor"] = prefetch
                kw["persistent_workers"] = False
            dl = DataLoader(Texts(), **kw)
            wait = 0.0
            t0 = time.perf_counter()
            it = iter(dl)
            while True:
                tw = time.perf_counter()
                try:
                    b = next(it)
                except StopIteration:
                    break
                wait += time.perf_counter() - tw
                b = {k: v.cuda(non_blocking=pin) for k, v in b.items()}
                enc(**b); _sync()
            total = time.perf_counter() - t0
            rows.append({"num_workers": workers, "pin_memory": pin, "prefetch_factor": prefetch,
                         "chunks_per_s": round(len(texts) / total, 1),
                         "data_wait_pct": round(100 * wait / total, 1), "seconds": round(total, 2)})
    return {"chunks": len(texts), "batch": 32, "cpu_count": os.cpu_count(), "rows": rows}


# ------------------------------------------------------------------------------------------------
# Chapter 4
# ------------------------------------------------------------------------------------------------
def _graphs():
    from experiments import _graph_count
    return _graph_count()


def exp_compile_stages():
    """Eager vs torch.compile (mode="default") for the BGE-M3 encoder and the Gemma-2B prefill."""
    import torch
    from sentence_transformers import SentenceTransformer
    out = {}
    m = SentenceTransformer(cfg.MODELS["embedder"], device="cuda", model_kwargs={"torch_dtype": _dtype()})
    tok = m.tokenizer
    batch = tok(_chunks(tok, 32), padding="max_length", truncation=True, max_length=cfg.EMBED_CHUNK_TOKENS,
                return_tensors="pt").to("cuda")
    enc = m[0].auto_model
    with torch.inference_mode():
        eager = _cuda_time(lambda: enc(**batch), warmup=2, iters=10)
        comp = torch.compile(enc, mode="default")
        t0 = time.perf_counter(); comp(**batch); _sync(); first = time.perf_counter() - t0
        compiled = _cuda_time(lambda: comp(**batch), warmup=2, iters=10)
    out["embedder_fp16_batch32"] = {"eager_ms": round(eager * 1000, 2), "compiled_ms": round(compiled * 1000, 2),
                                    "speedup": round(eager / compiled, 2), "first_call_s": round(first, 1)}
    del m, enc, comp; torch.cuda.empty_cache(); torch._dynamo.reset()

    gtok, model = _generator()
    inputs = _prompt(gtok)
    with torch.inference_mode():
        eager = _cuda_time(lambda: model(**inputs, use_cache=False), warmup=2, iters=10)
        comp = torch.compile(model, mode="default")
        t0 = time.perf_counter(); comp(**inputs, use_cache=False); _sync(); first = time.perf_counter() - t0
        compiled = _cuda_time(lambda: comp(**inputs, use_cache=False), warmup=2, iters=10)
    out["gemma_prefill_512"] = {"eager_ms": round(eager * 1000, 2), "compiled_ms": round(compiled * 1000, 2),
                                "speedup": round(eager / compiled, 2), "first_call_s": round(first, 1)}
    out["graphs_compiled"] = _graphs()
    return out


def _gpt2():
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(GPT2).cuda().eval()   # FP32, as in the chapter
    ids = torch.randint(0, model.config.vocab_size, (1, GPT2_SEQ), device="cuda")
    return model, ids


def exp_gpt2_explain():
    """The chapter's debugging walkthrough: GPT-2 with .item() + print, and the clean version."""
    import torch
    from torch import nn
    model, ids = _gpt2()

    class GPT2WithBreaks(nn.Module):
        def __init__(self, m):
            super().__init__(); self.model = m

        def forward(self, input_ids):
            embeds = self.model.transformer.wte(input_ids)
            max_val = embeds.abs().max().item()
            print(f"max magnitude: {max_val:.4f}")
            return self.model(inputs_embeds=embeds).logits

    class GPT2Clean(nn.Module):
        def __init__(self, m):
            super().__init__(); self.model = m

        def forward(self, input_ids):
            return self.model(input_ids).logits

    out = {"model": GPT2, "seq": GPT2_SEQ, "dtype": "float32"}
    with torch.inference_mode():
        for name, mod in (("with_breaks", GPT2WithBreaks(model)), ("clean", GPT2Clean(model))):
            torch._dynamo.reset()
            ex = torch._dynamo.explain(mod)(ids)
            out[name] = {"graph_count": ex.graph_count, "graph_break_count": ex.graph_break_count,
                         "op_count": ex.op_count, "text": str(ex)[:6000]}
    with open(_out("gpt2_explain.txt"), "w") as f:
        for name in ("with_breaks", "clean"):
            f.write(f"===== {name} =====\n{out[name]['text']}\n\n")
    return out


def _gpt2_first_and_steady(regional=False):
    import torch
    model, ids = _gpt2()
    with torch.inference_mode():
        eager = _cuda_time(lambda: model(ids), warmup=2, iters=10)
        if regional:
            for blk in model.transformer.h:
                blk.compile()
            fn = model
        else:
            fn = torch.compile(model)
        t0 = time.perf_counter(); fn(ids); _sync(); first = time.perf_counter() - t0
        steady = _cuda_time(lambda: fn(ids), warmup=2, iters=10)
    return {"model": GPT2, "seq": GPT2_SEQ, "dtype": "float32", "regional": regional,
            "inductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "eager_ms": round(eager * 1000, 2), "first_call_s": round(first, 2),
            "steady_ms": round(steady * 1000, 2), "first_over_steady": round(first / steady)}


def exp_gpt2_cold():
    return _gpt2_first_and_steady(regional=False)


def exp_gpt2_warm():
    return _gpt2_first_and_steady(regional=False)


def exp_gpt2_regional():
    return _gpt2_first_and_steady(regional=True)


def exp_gpt2_profile():
    """Eager vs compiled GPT-2 forward: key_averages tables and Chrome traces."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    model, ids = _gpt2()
    act = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    out = {"model": GPT2, "seq": GPT2_SEQ, "dtype": "float32"}
    with torch.inference_mode():
        for label, fn in (("eager", model), ("compiled", torch.compile(model))):
            for _ in range(3):
                fn(ids)
            _sync()
            with profile(activities=act) as prof:
                fn(ids); _sync()
            _save_trace(prof, f"gpt2_{label}"); _table(prof, f"gpt2_{label}", rows=12)
            ka = prof.key_averages()
            dev = lambda e: getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
            kern = [e for e in ka if dev(e) > 0 and not e.key.startswith("aten::")]
            out[label] = {"top_ops": _top_ops(prof, 12), "distinct_kernels": len(kern),
                          "kernel_launches": int(sum(e.count for e in kern)),
                          "total_cuda_ms": round(sum(dev(e) for e in kern) / 1000, 3)}
    return out


# ------------------------------------------------------------------------------------------------
# Chapter 5
# ------------------------------------------------------------------------------------------------
def exp_roofs():
    """Achievable roofs: large-GEMM TFLOP/s per precision, device-memory copy bandwidth, and the
    host-side cost of launching a tiny kernel."""
    import torch
    cc = torch.cuda.get_device_capability(0)
    n = 8192
    out = {"gemm_n": n, "gemm_tflops": {}}
    cases = [("fp32", torch.float32, False), ("tf32", torch.float32, True), ("fp16", torch.float16, False),
             ("bf16", torch.bfloat16, False)]
    for name, dt, tf32 in cases:
        if name in ("tf32", "bf16") and cc[0] < 8:
            out["gemm_tflops"][name] = None
            continue
        torch.backends.cuda.matmul.allow_tf32 = tf32
        a = torch.randn(n, n, device="cuda", dtype=dt); b = torch.randn(n, n, device="cuda", dtype=dt)
        t = _cuda_time(lambda: a @ b, warmup=3, iters=10)
        out["gemm_tflops"][name] = round(2 * n ** 3 / t / 1e12, 1)
        del a, b
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.empty_cache()
    x = torch.empty(1 << 28, device="cuda", dtype=torch.float32); y = torch.empty_like(x)   # 1 GiB each
    t = _cuda_time(lambda: y.copy_(x), warmup=3, iters=20)
    out["copy_gbs"] = round(2 * x.numel() * 4 / t / 1e9, 1)
    del x, y
    s = torch.ones(1, device="cuda")
    for _ in range(100):
        s.add_(1)
    _sync()
    k = 10000
    t0 = time.perf_counter()
    for _ in range(k):
        s.add_(1)
    enq = (time.perf_counter() - t0) / k
    _sync()
    t0 = time.perf_counter()
    for _ in range(1000):
        s.add_(1); _sync()
    rt = (time.perf_counter() - t0) / 1000
    out["launch_us"] = {"enqueue_per_kernel": round(enq * 1e6, 2), "launch_plus_sync": round(rt * 1e6, 2)}
    return out


# ------------------------------------------------------------------------------------------------
# Chapter 7 (multi-GPU container)
# ------------------------------------------------------------------------------------------------
def _allreduce_worker(rank, world, port, path):
    import torch
    import torch.distributed as dist
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    rows = []
    for mb in (1, 4, 16, 64, 256, 1024):
        nbytes = mb * (1 << 20)
        x = torch.ones(nbytes // 2, dtype=torch.float16, device="cuda")
        for _ in range(5):
            dist.all_reduce(x)
        torch.cuda.synchronize(); dist.barrier()
        iters = 20
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) / iters
        alg = nbytes / t / 1e9
        rows.append({"size_mb": mb, "time_ms": round(t * 1000, 3), "algbw_gbs": round(alg, 1),
                     "busbw_gbs": round(alg * 2 * (world - 1) / world, 1)})
    if rank == 0:
        json.dump(rows, open(path, "w"))
    dist.destroy_process_group()


def exp_allreduce():
    import torch
    import torch.multiprocessing as mp
    world = torch.cuda.device_count()
    path = _out("allreduce_rows.json")
    mp.spawn(_allreduce_worker, args=(world, 29511, path), nprocs=world, join=True)
    import subprocess
    topo = subprocess.getoutput("nvidia-smi topo -m")
    return {"gpus": world, "gpu": torch.cuda.get_device_name(0), "nccl": (".".join(map(str, v)) if isinstance((v := torch.cuda.nccl.version()), tuple) else str(v)),
            "rows": json.load(open(path)), "topology": topo[:4000]}


EXTRA = {
    "warmup": exp_warmup, "traces": exp_traces, "nsys_target": exp_nsys_target,
    "python_loop": exp_python_loop, "autograd": exp_autograd, "dataloader": exp_dataloader,
    "compile_stages": exp_compile_stages, "gpt2_explain": exp_gpt2_explain, "gpt2_cold": exp_gpt2_cold,
    "gpt2_warm": exp_gpt2_warm, "gpt2_regional": exp_gpt2_regional, "gpt2_profile": exp_gpt2_profile,
    "roofs": exp_roofs, "allreduce": exp_allreduce,
}

if __name__ == "__main__":
    name, out_path = sys.argv[1], sys.argv[2]
    json.dump(EXTRA[name](), open(out_path, "w"))
