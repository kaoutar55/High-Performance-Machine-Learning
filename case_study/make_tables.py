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
    return T


if __name__ == "__main__":
    T = build()
    (RESULTS / "master_tables.json").write_text(json.dumps(T, indent=2))
    (RESULTS / "master_tables.md").write_text(to_md(T))
    print(to_md(T))
