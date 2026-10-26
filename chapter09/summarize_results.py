"""Print the Chapter 9 case-study tables from chapter09/results/*.json.

    python chapter09/summarize_results.py > chapter09/results/summary.txt
"""
import json
import pathlib

RES = pathlib.Path(__file__).resolve().parent / "results"


def f(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def kernels(lines):
    keys = ("Using", "Selected", "attention backend", "does not have native", "Falling back")
    return [l.split("] ")[-1][:120] for l in lines if any(k in l for k in keys)][:4]


def perf():
    for p in sorted(RES.glob("*_*.json")):
        if p.name.startswith(("quality", "reference", "sparse", "build", "listings")) or "errors" in p.name:
            continue
        d = json.loads(p.read_text())
        print(f"\n=== {p.stem}")
        for v, r in d.get("perf", {}).items():
            st, s = r.get("startup", {}), r.get("single", {})
            if "error" in r and not st:
                print(f"  {v:20s} ERROR {r['error'][:200]}"); continue
            sweep = r.get("sweep", [])
            passed = [x["rate"] for x in sweep if x.get("slo_attainment", 0) >= 0.9]
            reps = r.get("knee", {}).get("repeats", [])
            knee = {}
            for rep in reps:
                for rate, x in rep["runs"].items():
                    knee.setdefault(rate, []).append(x.get("slo_attainment"))
            cap = max([float(k) for k, v_ in knee.items() if v_ and min(a or 0 for a in v_) >= 0.9] or [0])
            tput = {k: [x.get("output_token_throughput") for rep in reps for kk, x in rep["runs"].items() if kk == k]
                    for k in knee}
            print(f"  {v:20s} weights {f(st.get('weights_gib'), 2)} GiB  KV {st.get('kv_cache_tokens')} tok  "
                  f"TTFT p50 {f(s.get('ttft_ms', {}).get('p50'), 0)} ms  TPOT p50 {f(s.get('tpot_ms', {}).get('p50'), 2)} ms  "
                  f"sweep pass {passed}  capacity {cap:g} req/s")
            for k, a in knee.items():
                t = [round(x) for x in tput.get(k, []) if x]
                print(f"      knee {k:>5}: within {[round(100 * (x or 0), 1) for x in a]}  out tok/s {t}")
            for line in kernels(st.get("lines", [])):
                print(f"      {line}")
        for study in ("prefill", "long"):
            for v, r in d.get(study, {}).items():
                if "runs" not in r:
                    print(f"  {study} {v}: ERROR {r.get('error', '')[:200]}"); continue
                st = r.get("startup", {})
                print(f"  {study} {v:20s} KV {st.get('kv_cache_tokens')} tok")
                for x in r["runs"]:
                    if "ttft_ms" not in x:
                        continue
                    pk = x.get("metrics_peak", {})
                    kv = max([val for k, val in pk.items() if "cache_usage" in k] or [None]) if pk else None
                    pre = (sum(val for k, val in x.get("metrics_after", {}).items() if "preemption" in k)
                           - sum(val for k, val in x.get("metrics_before", {}).items() if "preemption" in k)) if pk else None
                    print(f"      rate {x['rate']:>5}: done {f(x.get('request_throughput'), 2)}/s  TTFT p50/p99 "
                          f"{f(x['ttft_ms']['p50'], 0)}/{f(x['ttft_ms']['p99'], 0)} ms  TPOT p50 "
                          f"{f(x.get('tpot_ms', {}).get('p50'), 1)}  within {f(100 * x.get('slo_attainment', 0))}%"
                          + (f"  KV {f(100 * kv if kv is not None else None, 0)}%  preempt {pre:.0f}" if pk else ""))


def quality():
    for m in ("llama", "gemma"):
        ref, q = load(f"reference_{m}.json"), load(f"quality_{m}.json")
        if not q:
            continue
        print(f"\n=== quality {m}  (reference noise floor: {ref and ref.get('noise_floor')})")
        print("  variant               KL mean   KL p99   top-1   same answer  lm-eval")
        for v, r in q.get("variants", {}).items():
            if "error" in r and "kl" not in r:
                print(f"  {v:20s} ERROR {r['error'][:200]}"); continue
            kl, g = r.get("kl", {}), r.get("generation", {})
            mets = r.get("lm_eval", {}).get("metrics", {})
            short = []
            for task, ms in mets.items():
                key = next((k for k in ms if k.startswith(("exact_match,flexible", "exact_match,strict",
                                                           "prompt_level_strict_acc", "acc,none", "word_perplexity"))
                            and "stderr" not in k), None)
                if key:
                    short.append(f"{task}={ms[key]:.4g}")
            print(f"  {v:20s} {f(kl.get('kl_mean'), 4):>8} {f(kl.get('kl_p99'), 3):>8} {f(kl.get('top1_agreement'), 3):>7}"
                  f" {f(g.get('identical_answers'), 3):>11}  {' '.join(short)}")
            if kl.get("method", "").startswith("top-k"):
                print(f"      KL method: {kl['method'][:160]}")


def other():
    b = load("build_info.json")
    if b:
        print("\n=== checkpoints built")
        for r in b:
            print(f"  {r.get('model', '')[-28:]:28s} {r.get('variant', ''):22s} "
                  f"{r.get('error', '')[:150] or str(r.get('size_gb')) + ' GB, ' + str(r.get('build_s')) + ' s'}"
                  + (f", sparsity {r['linear_sparsity']:.3f} valid 2:4 {r['valid_2_4']}" if "linear_sparsity" in r else ""))
    for tag in ("A100", "H100"):
        g = load(f"sparse_gemm_{tag}.json")
        if not g or "rows" not in g:
            continue
        print(f"\n=== 2:4 GEMM {g['gpu']} (torch {g['torch']}): speedup over dense FP16")
        for r in g["rows"]:
            if "tokens" in r:
                sp = lambda k: f"{r[k]['speedup']:.2f}" if "speedup" in r.get(k, {}) else "-"
                print(f"  {r['layer']:13s} tokens {r['tokens']:6d}  dense {r['dense'].get('ms', 0):.4f} ms  "
                      f"cuSPARSELt {sp('cusparselt')}  CUTLASS {sp('cutlass')}")
    li = load("listings.json")
    if li:
        print("\n=== listings")
        for k, v in li.items():
            print(f"-- {k}\n{v['stdout'] if isinstance(v, dict) else v}")


if __name__ == "__main__":
    other(); quality(); perf()
