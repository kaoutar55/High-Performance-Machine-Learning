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


def decode_is_bandwidth_limited(res):
    """At batch 1 a decode step reads every weight once to produce one token per sequence, about 1 FLOP
    per byte, so it can use only a tiny fraction of peak compute; its floor is memory bandwidth."""
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
    return pct_compute < 5, (f"{row['decode_ms_per_step']:.1f} ms/token = {pct_compute:.2f}% of peak compute; "
                             f"bandwidth floor {floor_ms:.1f} ms/token, so the step reaches {pct_bw:.0f}% of the "
                             f"bandwidth limit (the rest is framework overhead)")


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
    """In the naive RAG pipeline, token-by-token decode dominates end-to-end latency, so it is where
    optimization starts."""
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


def batching_feeds_the_encoder(res):
    """Batching the embedder keeps the GPU busy: throughput rises steeply from batch 1 until the GPU
    saturates."""
    rows = _get(res, "embedder", "throughput")
    b1, b32 = _row(rows, dtype="fp16", batch=1), _row(rows, dtype="fp16", batch=32)
    if not b1 or not b32:
        return None, "not measured"
    gain = b32["chunks_per_s"] / b1["chunks_per_s"]
    return gain >= 3, f"FP16 batch 32 gives {gain:.1f}x the chunks/s of batch 1"


def compile_only_helps_compiled_code(res):
    """torch.compile speeds up only the code that actually runs through the compiled function:
    compiling the module and then calling .generate() leaves decode unchanged; compiling forward
    with a static KV cache speeds it up."""
    c = res.get("compiled_decode") or {}
    eager = _get(c, "eager_static_cache", "decode_ms_per_token")
    eager_dyn = _get(c, "eager_dynamic_cache", "decode_ms_per_token")
    wrong = _get(c, "wrong_compile_module_then_generate", "decode_ms_per_token")
    right = _get(c, "right_compile_forward_static_cache", "decode_ms_per_token")
    if None in (eager, eager_dyn, wrong, right):
        return None, "not measured (see compiled_decode errors)"
    unchanged = abs(wrong - eager_dyn) / eager_dyn < 0.10
    faster = right < 0.9 * eager
    return unchanged and faster, (f"eager {eager_dyn:.1f} ms/token, compile-module-then-generate {wrong:.1f}, "
                                  f"compiled forward + static cache {right:.1f} (eager static {eager:.1f})")


def attention_kernel_depends_on_generation(res):
    """Which fused attention kernel runs depends on the GPU generation: FlashAttention needs compute
    capability 8.0 or newer; older GPUs fall back to another kernel."""
    seen = _get(res, "embedder", "sdpa_backend_seen")
    cc = _get(res, "env", "compute_capability")
    if not seen or not cc:
        return None, "not measured"
    major = int(str(cc).split(".")[0])
    expected_flash = major >= 8
    return seen.get("flash") == expected_flash, f"compute capability {cc}; flash kernel seen: {seen.get('flash')}"


PRINCIPLES = [
    ("decode-bandwidth", "Decode at batch 1 is limited by memory bandwidth, not compute", "Ch 1, 2, 5", decode_is_bandwidth_limited),
    ("batching", "Batching amortizes weight reads during decode", "Ch 5", batching_amortizes_weights),
    ("prefill-compute", "Prefill is compute-heavy; decode is not", "Ch 1, 5", prefill_is_compute_heavy),
    ("dominant-stage", "Decode dominates the naive RAG pipeline's latency", "Ch 1, 2", decode_dominates_naive_rag),
    ("precision", "16-bit Tensor Core math speeds up the encoder", "Ch 3, 5", lower_precision_speeds_up_encoder),
    ("encoder-batching", "Batching keeps the encoder's GPU busy", "Ch 3", batching_feeds_the_encoder),
    ("compile-scope", "torch.compile helps only code that runs through the compiled function", "Ch 4", compile_only_helps_compiled_code),
    ("attention-kernel", "The fused attention kernel depends on the GPU generation", "Ch 2, 3, 5", attention_kernel_depends_on_generation),
]


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
