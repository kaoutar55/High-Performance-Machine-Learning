"""Build the book's case-study tables from results/<gpu>.json.

Every number the chapters quote from the case study comes from this one file, so a re-run updates
the manuscript mechanically:
    python case_study/make_tables.py            # writes case_study/results/master_tables.md and .json
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))
import experiment_config as cfg  # noqa: E402


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def row(rows, **m):
    return next((r for r in rows or [] if all(r.get(k) == v for k, v in m.items())), None)


def fmt(x, nd=1):
    return "–" if x is None else f"{x:,.{nd}f}"


def load(results=RESULTS):
    out = {}
    for gpu in cfg.GPUS:
        p = pathlib.Path(results) / f"{gpu.lower().replace('-', '_')}.json"
        if p.exists():
            out[gpu] = json.loads(p.read_text())
    return out


def load_spread(results=RESULTS):
    """{gpu: (min_result, max_result)} from <gpu>_spread.json, when repeats were run."""
    out = {}
    for gpu in cfg.GPUS:
        p = pathlib.Path(results) / f"{gpu.lower().replace('-', '_')}_spread.json"
        if p.exists():
            sp = json.loads(p.read_text())
            pick = lambda x, i: ({k: pick(v, i) for k, v in x.items()} if isinstance(x, dict) else
                                 x[i] if isinstance(x, list) and len(x) == 2 and all(
                                     isinstance(v, (int, float)) for v in x) else
                                 [pick(v, i) for v in x] if isinstance(x, list) else x)
            out[gpu] = (pick(sp, 0), pick(sp, 1))
    return out


def _num(cell):
    try:
        return float(str(cell).replace(",", "").rstrip("%"))
    except ValueError:
        return None


def with_ranges(T, T_lo, T_hi, threshold=0.10):
    """Add "(min–max)" to every numeric cell whose spread across repeats exceeds threshold."""
    for name, t in T.items():
        for i, row_ in enumerate(t["rows"]):
            for j in range(1, len(row_)):
                m, lo, hi = _num(row_[j]), _num(T_lo[name]["rows"][i][j]), _num(T_hi[name]["rows"][i][j])
                if None not in (m, lo, hi) and m and (hi - lo) / abs(m) > threshold:
                    t["rows"][i][j] = f"{row_[j]} ({T_lo[name]['rows'][i][j]}–{T_hi[name]['rows'][i][j]})"
        t.setdefault("note", "")
    return T


def tables(by):
    gpus = list(by)
    T = {}

    def add(name, header, rows, note=""):
        T[name] = {"header": header, "rows": rows, "note": note}

    add("platforms", ["", *gpus], [
        ["GPU", *[g(r, "env", "gpu") for r in by.values()]],
        ["Compute capability", *[g(r, "env", "compute_capability") for r in by.values()]],
        ["Host CPU", *[g(r, "env", "cpu") or "not recorded" for r in by.values()]],
        ["PyTorch / CUDA", *[f"{g(r, 'env', 'torch')} / {g(r, 'env', 'cuda')}" for r in by.values()]],
        ["transformers", *[g(r, "env", "transformers") for r in by.values()]],
    ])
    lab = {k: g(r, "chapter02_lab", "summary") or {} for k, r in by.items()}
    add("ch2_latency_ms", ["Statistic", *gpus],
        [[s, *[fmt(g(lab[k], "latency_ms", s)) for k in gpus]] for s in ("p50", "p95", "p99", "mean")],
        "Chapter 2 lab, naive RAG pipeline, 100 requests.")
    stages = list((g(next(iter(lab.values())), "stage_median_ms") or {}).keys())
    srows = [[s, *[fmt(g(lab[k], "stage_median_ms", s)) for k in gpus]] for s in stages]
    tot = {k: sum((g(lab[k], "stage_median_ms") or {}).values()) for k in gpus}
    srows.append(["Total of stage medians", *[fmt(tot[k]) for k in gpus]])
    dk = "decode" if "decode" in stages else "prefill_plus_decode"
    srows.append(["Decode share", *[f"{100 * g(lab[k], 'stage_median_ms', dk) / tot[k]:.0f}%" if tot[k] else "–"
                                    for k in gpus]])
    add("ch2_stage_median_ms", ["Stage", *gpus], srows, "Chapter 2 lab, median per stage.")
    add("ch2_decode_roofline", ["Quantity", *gpus], [
        ["Decode ms/token (lab loop)", *[fmt(g(lab[k], "decode_roofline", "ms_per_token")) for k in gpus]],
        ["Achieved bandwidth (GB/s)", *[fmt(g(lab[k], "decode_roofline", "achieved_bw_gbs"), 0) for k in gpus]],
        ["% of peak bandwidth", *[fmt(g(lab[k], "decode_roofline", "pct_peak_bw"), 0) for k in gpus]],
        ["Arithmetic intensity (FLOP/byte)", *[fmt(g(lab[k], "decode_roofline", "arithmetic_intensity")) for k in gpus]],
        ["Ridge point (FLOP/byte)", *[fmt(g(lab[k], "decode_roofline", "ridge"), 0) for k in gpus]],
    ], "The lab's hand-written decode loop runs more Python per token than generate().")
    emb = {k: g(r, "embedder") or {} for k, r in by.items()}
    erows = []
    for dt in ("fp32", "fp16"):
        for b in cfg.EMBED_BATCHES:
            erows.append([f"{dt.upper()}, batch {b}",
                          *[fmt(g(row(emb[k].get("throughput"), dtype=dt, batch=b) or {}, "chunks_per_s")) for k in gpus]])
    erows.append(["Attention kernel", *[emb[k].get("attention_backend") or
                                        ("flash" if g(emb[k], "sdpa_backend_seen", "flash") else "mem_efficient")
                                        for k in gpus]])
    add("ch3_embedder_chunks_per_s", ["Setting", *gpus], erows,
        f"BGE-M3 on {g(next(iter(emb.values())), 'tokens_per_chunk')}-token chunks.")
    cd = {k: g(r, "compiled_decode") or {} for k, r in by.items()}
    variants = [("eager_dynamic_cache", "Eager, dynamic cache"),
                ("eager_static_cache", "Eager, static cache (auto-compile off)"),
                ("hf_autocompile_static_cache", "Static cache, transformers auto-compile"),
                ("wrong_compile_module_then_generate", "torch.compile(model) then .generate()"),
                ("right_compile_reduce_overhead", "Compiled forward + static cache, CUDA graphs"),
                ("right_compile_default", "Compiled forward + static cache, no CUDA graphs")]
    add("ch4_decode_ms_per_token", ["Variant", *gpus],
        [[lab_, *[fmt(g(cd[k], v, "decode_ms_per_token"), 2) for k in gpus]] for v, lab_ in variants],
        "generate(), batch 1, 512-token prompt, 20 new tokens. '–' = not measured or crashed.")
    db = {k: g(r, "decode_batching") or {} for k, r in by.items()}
    brows = []
    for b in cfg.DECODE_BATCHES:
        brows.append([f"Batch {b}: decode tokens/s",
                      *[fmt(g(row(db[k].get("rows"), batch=b) or {}, "decode_tokens_per_s"), 0) for k in gpus]])
    for b in cfg.DECODE_BATCHES:
        brows.append([f"Batch {b}: ms per decode step",
                      *[fmt(g(row(db[k].get("rows"), batch=b) or {}, "decode_ms_per_step"), 2) for k in gpus]])
    brows.append(["Batch 1: prefill TFLOP/s", *[fmt(g(row(db[k].get("rows"), batch=1) or {}, "prefill_tflops")) for k in gpus]])
    brows.append(["Weight-read floor (ms/token)", *[fmt(db[k].get("weight_gb", 0) / (g(by[k], "spec", "mem_bw_gbs") or 1) * 1000, 2)
                                                     for k in gpus]])
    add("ch5_decode_batching", ["Quantity", *gpus], brows)

    ex = {k: g(r, "extra") or {} for k, r in by.items()}
    if any(ex.values()):
        def ev(k, *path, nd=1):
            return fmt(g(ex[k], *path), nd) if isinstance(g(ex[k], *path), (int, float)) else "–"
        w = {k: g(ex[k], "warmup", "call_us") or [] for k in gpus}
        add("ch2_warmup_us", ["Call", *gpus],
            [[f"Call {i + 1}", *[fmt(w[k][i]) if len(w[k]) > i else "–" for k in gpus]] for i in range(6)]
            + [["First / steady", *[ev(k, "warmup", "first_over_steady") for k in gpus]]],
            "torch.bmm on 32x512x512 FP32 in a fresh process, microseconds per call.")
        keys = []
        for k in gpus:
            for kk in (g(ex[k], "python_loop", "ms") or {}):
                if kk not in keys:
                    keys.append(kk)
        add("ch3_python_loop_ms", ["Method", *gpus],
            [[kk, *[ev(k, "python_loop", "ms", kk, nd=2) for k in gpus]] for kk in keys],
            "Summing 10 million floats (ms).")
        add("ch3_autograd", ["Mode", *gpus],
            [[f"{m}: ms", *[ev(k, "autograd", m, "ms", nd=2) for k in gpus]] for m in ("grad_enabled", "no_grad", "inference_mode")]
            + [[f"{m}: extra memory (GB)", *[ev(k, "autograd", m, "peak_extra_gb", nd=2) for k in gpus]]
               for m in ("grad_enabled", "no_grad", "inference_mode")],
            "Gemma-2B prefill, 512 tokens, FP16.")
        dl = {k: g(ex[k], "dataloader", "rows") or [] for k in gpus}
        n = max((len(v) for v in dl.values()), default=0)
        ref = next((v for v in dl.values() if v), [])
        lab_ = lambda r: f"workers={r['num_workers']}, pin={r['pin_memory']}, prefetch={r['prefetch_factor']}"
        add("ch3_dataloader", ["Setting", *gpus],
            [[lab_(ref[i]), *[(f"{fmt(dl[k][i]['chunks_per_s'])} chunks/s, {fmt(dl[k][i]['data_wait_pct'])}% waiting"
                               if len(dl[k]) > i else "–") for k in gpus]] for i in range(n)],
            "BGE-M3 FP16, batch 32, tokenization inside __getitem__.")
        rows_ = []
        for st, lab2 in (("embedder_fp16_batch32", "BGE-M3 encoder, batch 32"), ("gemma_prefill_512", "Gemma-2B prefill, 512 tokens")):
            for f, l in (("eager_ms", "eager ms"), ("compiled_ms", "compiled ms"), ("speedup", "speedup"), ("first_call_s", "first call s")):
                rows_.append([f"{lab2}: {l}", *[ev(k, "compile_stages", st, f, nd=2) for k in gpus]])
        add("ch4_compile_stages", ["Quantity", *gpus], rows_, "torch.compile(mode='default').")
        rows_ = []
        for e, lab2 in (("gpt2_cold", "full model, cold cache"), ("gpt2_warm", "full model, warm cache"),
                        ("gpt2_regional", "regional, cold cache")):
            rows_ += [[f"{lab2}: first call (s)", *[ev(k, e, "first_call_s", nd=2) for k in gpus]],
                      [f"{lab2}: steady call (ms)", *[ev(k, e, "steady_ms", nd=2) for k in gpus]]]
        rows_ += [["eager call (ms)", *[ev(k, "gpt2_cold", "eager_ms", nd=2) for k in gpus]]]
        for lab3 in ("with_breaks", "clean"):
            rows_ += [[f"explain {lab3}: graphs / breaks / ops",
                       *[" / ".join(str(g(ex[k], "gpt2_explain", lab3, f)) for f in ("graph_count", "graph_break_count", "op_count"))
                         for k in gpus]]]
        for lab3 in ("eager", "compiled"):
            rows_ += [[f"profile {lab3}: CUDA ms / kernels / launches",
                       *[" / ".join(str(g(ex[k], "gpt2_profile", lab3, f)) for f in ("total_cuda_ms", "distinct_kernels", "kernel_launches"))
                         for k in gpus]]]
        add("ch4_gpt2", ["Quantity", *gpus], rows_, f"GPT-2 124M, FP32, one sequence of 512 tokens.")
        rows_ = [[f"GEMM {p} (TFLOP/s)", *[ev(k, "roofs", "gemm_tflops", p) for k in gpus]] for p in ("fp32", "tf32", "fp16", "bf16")]
        rows_ += [["Copy bandwidth (GB/s)", *[ev(k, "roofs", "copy_gbs", nd=0) for k in gpus]],
                  ["Datasheet bandwidth (GB/s)", *[fmt(g(by[k], "spec", "mem_bw_gbs"), 0) for k in gpus]],
                  ["Datasheet FP16 Tensor (TFLOP/s)", *[fmt(g(by[k], "spec", "fp16_tensor_tflops"), 0) for k in gpus]],
                  ["Kernel enqueue (us)", *[ev(k, "roofs", "launch_us", "enqueue_per_kernel", nd=2) for k in gpus]],
                  ["Launch + sync round trip (us)", *[ev(k, "roofs", "launch_us", "launch_plus_sync", nd=2) for k in gpus]]]
        add("ch5_measured_roofs", ["Quantity", *gpus], rows_, "8192^3 GEMM; 1 GiB device-to-device copy; tiny add_ kernel.")
    return T


def to_md(T):
    out = ["# Case-study master tables", "", "Generated by case_study/make_tables.py from results/<gpu>.json.", ""]
    for name, t in T.items():
        out += [f"## {name}", ""]
        if t["note"]:
            out += [t["note"], ""]
        out.append("| " + " | ".join(t["header"]) + " |")
        out.append("|" + "---|" * len(t["header"]))
        out += ["| " + " | ".join(str(c) for c in r) + " |" for r in t["rows"]]
        out.append("")
    return "\n".join(out)


def build(results=RESULTS):
    by = load(results)
    if not by:
        raise SystemExit(f"No results in {results}")
    T = tables(by)
    sp = load_spread(results)
    if sp and set(sp) == set(by):
        T = with_ranges(T, tables({g: sp[g][0] for g in by}), tables({g: sp[g][1] for g in by}))
        n = {g: by[g].get("repeats", 1) for g in by}
        for t in T.values():
            t["note"] = (t["note"] + " " if t["note"] else "") + (
                f"Median of {min(n.values()) if min(n.values()) == max(n.values()) else f'{min(n.values())}–{max(n.values())}'} "
                "independent runs; (min–max) shown where runs "
                "differ by more than 10%.")
    ar = sorted(pathlib.Path(results).glob("allreduce_*.json"))
    if ar:
        runs = {p.stem.removeprefix("allreduce_"): json.loads(p.read_text()) for p in ar}
        sizes = [r["size_mb"] for r in next(iter(runs.values()))["allreduce"].get("rows", [])]
        T["ch7_allreduce_busbw_gbs"] = {
            "header": ["Message size", *[f"{g(v, 'allreduce', 'gpus')}x {g(v, 'env', 'gpu')}" for v in runs.values()]],
            "rows": [[f"{mb} MB", *[fmt(next((r["busbw_gbs"] for r in g(v, "allreduce", "rows") or [] if r["size_mb"] == mb), None))
                                    for v in runs.values()]] for mb in sizes],
            "note": "NCCL all-reduce of FP16 tensors across all GPUs in one container; bus bandwidth = "
                    "algorithm bandwidth x 2(n-1)/n (nccl-tests convention)."}
    return T


if __name__ == "__main__":
    T = build()
    (RESULTS / "master_tables.json").write_text(json.dumps(T, indent=2))
    (RESULTS / "master_tables.md").write_text(to_md(T))
    print(to_md(T))
