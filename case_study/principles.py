"""The durable claims the book teaches, each checked against every GPU's measurements.

The numbers in a results file belong to one GPU, one model and one software stack, and they will
change. These principles should not: each one states the claim in plain words, names the chapters
that teach it, and checks it against the measured results. A principle that fails on a new GPU is
a finding worth a sentence in the book, not something to hide.

Each check returns (holds, evidence): holds is True, False or None (not measurable on this run).
"""


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _row(rows, **match):
    for r in rows or []:
        if all(r.get(k) == v for k, v in match.items()):
            return r
    return None


def decode_ceiling_is_bandwidth(res):
    """A decode step at batch 1 reads every weight once to produce one token, about 1 FLOP per byte,
    so its ceiling is memory bandwidth: it can use only a tiny fraction of peak compute."""
    spec, row = res.get("spec"), _row(_get(res, "decode_batching", "rows"), batch=1)
    if not spec or not row or "decode_ms_per_step" not in row:
        return None, "not measured"
    params = _get(res, "decode_batching", "params_b") * 1e9
    weight_gb = _get(res, "decode_batching", "weight_gb")
    step_s = row["decode_ms_per_step"] / 1000
    tflops = 2 * params / step_s / 1e12
    floor_ms = weight_gb / spec["mem_bw_gbs"] * 1000
    pct_compute = 100 * tflops / spec["fp16_tensor_tflops"]
    pct_bw = 100 * floor_ms / row["decode_ms_per_step"]
    return pct_compute < 5, (f"{row['decode_ms_per_step']:.1f} ms/token uses {pct_compute:.2f}% of peak compute; "
                             f"weight-read floor {floor_ms:.1f} ms/token, reached at {pct_bw:.0f}%")


def overhead_decides_how_close_decode_gets(res):
    """Whether decode reaches its bandwidth ceiling depends on per-step overhead (Python, kernel
    launches). Removing that overhead (static cache + compilation / CUDA graphs) helps in proportion
    to the overhead's share of the step: a lot on fast GPUs, little on a GPU already near its floor."""
    spec, c = res.get("spec"), res.get("compiled_decode") or {}
    weight_gb = _get(res, "decode_batching", "weight_gb")
    eager = _get(c, "eager_dynamic_cache", "decode_ms_per_token")
    cands = [(k, _get(c, k, "decode_ms_per_token")) for k in
             ("hf_autocompile_static_cache", "right_compile_forward_static_cache")
             if _get(c, k, "decode_ms_per_token") and (_get(c, k, "dynamo_graphs_compiled") or 0) > 0]
    if not spec or not weight_gb or not eager or not cands:
        return None, "not measured (no compiled decode number)"
    # Book-wide choice: quote transformers' own static-cache compilation when it ran; otherwise the
    # hand-compiled forward.
    label, fast = next((kv for kv in cands if kv[0] == "hf_autocompile_static_cache"), cands[0])
    floor = weight_gb / spec["mem_bw_gbs"] * 1000
    share = 1 - floor / eager
    speedup = eager / fast
    holds = (speedup >= 1.3) if share > 0.5 else (speedup < 1.3)
    return holds, (f"eager {eager:.1f} ms/token, floor {floor:.1f} ms, so overhead is {100 * share:.0f}% of the step; "
                   f"{label} gives {fast:.1f} ms/token ({speedup:.2f}x)")


def batching_amortizes_weights(res):
    """Batching amortizes weight reads: decode throughput grows almost in proportion to batch size
    until compute or KV-cache traffic takes over."""
    rows = _get(res, "decode_batching", "rows")
    b1, b16 = _row(rows, batch=1), _row(rows, batch=16)
    if not b1 or not b16 or "decode_tokens_per_s" not in b16:
        return None, "not measured"
    gain = b16["decode_tokens_per_s"] / b1["decode_tokens_per_s"]
    return gain >= 8, f"batch 16 gives {gain:.1f}x the tokens/s of batch 1 (ideal 16x)"


def prefill_is_compute_heavy(res):
    """Prefill processes every prompt token per weight read, so it runs far closer to peak compute
    than decode does."""
    rows = _get(res, "decode_batching", "rows")
    b1 = _row(rows, batch=1)
    if not b1 or "prefill_tflops" not in b1 or "decode_ms_per_step" not in b1:
        return None, "not measured"
    params = _get(res, "decode_batching", "params_b") * 1e9
    dec_tflops = 2 * params / (b1["decode_ms_per_step"] / 1000) / 1e12
    ratio = b1["prefill_tflops"] / dec_tflops
    return ratio >= 10, f"prefill {b1['prefill_tflops']:.1f} TFLOP/s vs decode {dec_tflops:.2f} TFLOP/s ({ratio:.0f}x)"


def decode_dominates_naive_rag(res):
    """In this case study (small in-memory index, one retrieval, 2B generator, 20 output tokens),
    token-by-token decode dominates end-to-end latency. At billion-scale retrieval or with long
    contexts encoded online, retrieval or encoding can dominate instead (RAGO, Shen et al.)."""
    stages = _get(res, "chapter02_lab", "summary", "stage_median_ms")
    key = next((k for k in ("decode", "prefill_plus_decode") if k in (stages or {})), None)
    if not key:
        return None, "not measured"
    total = sum(stages.values())
    share = 100 * stages[key] / total
    return share > 50, f"{key} {stages[key]:.0f} ms of {total:.0f} ms stage total ({share:.0f}%)"


def lower_precision_speeds_up_encoder(res):
    """Running the encoder in 16-bit on Tensor Cores raises throughput substantially over FP32."""
    rows = _get(res, "embedder", "throughput")
    f16, f32 = _row(rows, dtype="fp16", batch=32), _row(rows, dtype="fp32", batch=32)
    if not f16 or not f32:
        return None, "not measured"
    gain = f16["chunks_per_s"] / f32["chunks_per_s"]
    return gain >= 1.5, f"FP16 {f16['chunks_per_s']:.0f} vs FP32 {f32['chunks_per_s']:.0f} chunks/s at batch 32 ({gain:.1f}x)"


def rows_per_gemm_set_utilization(res):
    """GPU utilization depends on how many rows each matrix multiply processes. Decode has one row per
    sequence, so batching sequences helps almost linearly. An embedder chunk already supplies hundreds
    of token rows, so batching chunks helps less; how much less depends on where the GPU's knee is."""
    dec = _get(res, "decode_batching", "rows")
    emb = _get(res, "embedder", "throughput")
    d1, d16 = _row(dec, batch=1), _row(dec, batch=16)
    e1, e32 = _row(emb, dtype="fp16", batch=1), _row(emb, dtype="fp16", batch=32)
    rows = _get(res, "embedder", "tokens_per_chunk")
    if not all((d1, d16, e1, e32)) or "decode_tokens_per_s" not in d16:
        return None, "not measured"
    dgain = d16["decode_tokens_per_s"] / d1["decode_tokens_per_s"]
    egain = e32["chunks_per_s"] / e1["chunks_per_s"]
    return dgain > egain, (f"decode (1 row/sequence): batch 16 = {dgain:.1f}x batch 1; "
                           f"embedder ({rows} token rows/chunk): batch 32 = {egain:.1f}x batch 1")


def compile_only_helps_compiled_code(res):
    """torch.compile speeds up only code that runs through the compiled function: compiling the module
    and then calling .generate() leaves the decode loop, which generate() drives, essentially unchanged."""
    c = res.get("compiled_decode") or {}
    eager = _get(c, "eager_dynamic_cache", "decode_ms_per_token")
    wrong = _get(c, "wrong_compile_module_then_generate", "decode_ms_per_token")
    if None in (eager, wrong):
        return None, "not measured"
    return abs(wrong - eager) / eager < 0.15, f"eager {eager:.1f} ms/token; compile-module-then-generate {wrong:.1f}"


def attention_runs_fused(res):
    """Attention runs as one fused kernel on every GPU; which kernel PyTorch's SDPA dispatcher picks
    (FlashAttention, memory-efficient, cuDNN) depends on GPU, dtype, mask and library versions, so read
    it from the profiler."""
    be = _get(res, "embedder", "attention_backend")
    if be is None:
        seen = _get(res, "embedder", "sdpa_backend_seen")
        if not seen:
            return None, "not measured"
        be = "flash" if seen.get("flash") else "mem_efficient" if seen.get("mem_efficient") else "math (unfused)"
    ops = (_get(res, "embedder", "attention_ops") or [""])
    cc = _get(res, "env", "compute_capability")
    return be != "math (unfused)", f"compute capability {cc}: {be} ({ops[-1][:60]})"


PRINCIPLES = [
    ("decode-ceiling", "At batch 1 every decode GEMM is a matrix-vector product (~1 FLOP/byte), so memory bandwidth sets the Roofline ceiling", "Ch 1, 2, 5", decode_ceiling_is_bandwidth),
    ("decode-overhead", "Eager decode pays a roughly fixed CPU and launch cost per step; removing it helps in proportion to that cost's share of the step", "Ch 2, 3, 4", overhead_decides_how_close_decode_gets),
    ("batching", "A decode batch reads the weights once per step, so tokens/s rises almost linearly with batch size until compute or KV-cache traffic takes over", "Ch 5", batching_amortizes_weights),
    ("prefill-compute", "Prefill is compute-bound; decode is memory-bound", "Ch 1, 5", prefill_is_compute_heavy),
    ("rows-per-gemm", "Utilization grows with the rows per matrix multiply until a hardware-dependent knee", "Ch 1, 3, 5", rows_per_gemm_set_utilization),
    ("dominant-stage", "In this case study's configuration, decode dominates the RAG pipeline's latency", "Ch 1, 2", decode_dominates_naive_rag),
    ("precision", "16-bit Tensor Core math gives several times FP32 throughput on compute-bound work", "Ch 3, 5", lower_precision_speeds_up_encoder),
    ("compile-scope", "torch.compile(model) compiles forward only; generate()'s loop stays eager", "Ch 4", compile_only_helps_compiled_code),
    ("attention-fused", "SDPA dispatches at run time to the first eligible fused attention backend", "Ch 2, 3, 5", attention_runs_fused),
]

# Where each principle is taught in the literature (checked 23 Oct 2026; full quotes in the book's
# analysis/principles_sources.md). Cite these in the chapters next to the measured evidence.
SOURCES = {
    "decode-ceiling": ["Williams, Waterman, Patterson, Roofline, CACM 2009",
                       "NVIDIA, GPU Performance Background User's Guide"],
    "decode-overhead": ["PyTorch blog, Accelerating Generative AI with PyTorch II: GPT, Fast (2023)",
                        "Ansel et al., PyTorch 2, ASPLOS 2024"],
    "batching": ["Agrawal et al., Sarathi-Serve, OSDI 2024", "Pope et al., Efficiently Scaling Transformer Inference, MLSys 2023",
                 "Kwon et al., PagedAttention, SOSP 2023"],
    "prefill-compute": ["Agrawal et al., Sarathi-Serve, OSDI 2024", "Patel et al., Splitwise, ISCA 2024"],
    "rows-per-gemm": ["NVIDIA, Linear/Fully-Connected Layers User's Guide", "Agrawal et al., Sarathi-Serve, OSDI 2024"],
    "dominant-stage": ["Shen et al., arXiv:2412.11854 (retrieval can be 41% of latency)",
                       "Jiang et al., RAGO, arXiv:2503.14649 (the dominant stage depends on configuration)"],
    "precision": ["Micikevicius et al., Mixed Precision Training, ICLR 2018", "PyTorch CUDA semantics: TF32 notes"],
    "compile-scope": ["Hugging Face, LLM optimization overview (compile inside generate)",
                      "PyTorch blog, GPT, Fast (compile the per-token decode step)"],
    "attention-fused": ["Dao et al., FlashAttention, NeurIPS 2022", "PyTorch 2.5 release blog (cuDNN SDPA on H100)"],
}


def evaluate(results_by_gpu):
    """Return rows: principle id, statement, chapters, and per-GPU (holds, evidence)."""
    out = []
    for pid, statement, chapters, fn in PRINCIPLES:
        per_gpu = {}
        for gpu, res in results_by_gpu.items():
            try:
                per_gpu[gpu] = fn(res)
            except Exception as e:  # noqa: BLE001
                per_gpu[gpu] = (None, f"check failed: {e!r}"[:200])
        out.append({"id": pid, "statement": statement, "chapters": chapters, "why": fn.__doc__.strip(),
                    "sources": SOURCES.get(pid, []),
                    "results": {g: {"holds": h, "evidence": ev} for g, (h, ev) in per_gpu.items()}})
    return out


def markdown_report(rows, results_by_gpu):
    gpus = list(results_by_gpu)
    mark = {True: "holds", False: "FAILS", None: "n/a"}
    lines = ["# Principles across GPUs", "",
             "Each row is a claim the book teaches. The evidence comes from this run's measurements.", ""]
    lines.append("| Principle | Chapters | " + " | ".join(gpus) + " |")
    lines.append("|---|---|" + "---|" * len(gpus))
    for r in rows:
        cells = [mark[r["results"][g]["holds"]] for g in gpus]
        lines.append(f"| {r['statement']} | {r['chapters']} | " + " | ".join(cells) + " |")
    lines += ["", "## Evidence", ""]
    for r in rows:
        lines.append(f"### {r['statement']}")
        lines.append(f"_{r['why']}_")
        if r.get("sources"):
            lines.append("")
            lines.append("Sources: " + "; ".join(r["sources"]))
        lines.append("")
        for g in gpus:
            lines.append(f"- **{g}** ({mark[r['results'][g]['holds']]}): {r['results'][g]['evidence']}")
        lines.append("")
    lines += ["## Platforms", ""]
    for g, res in results_by_gpu.items():
        env = res.get("env") or {}
        lines.append(f"- **{g}**: {env.get('gpu', '?')} (CC {env.get('compute_capability', '?')}), "
                     f"PyTorch {env.get('torch', '?')}, CUDA {env.get('cuda', '?')}, "
                     f"transformers {env.get('transformers', '?')}")
    return "\n".join(lines) + "\n"
