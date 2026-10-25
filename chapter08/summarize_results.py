"""Print the Chapter 8 case-study tables from results/<gpu>.json and plot the load sweep.

    python chapter08/summarize_results.py            # tables to stdout, results/sweep.png
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"


def f(x, nd=0):
    return "—" if x is None else f"{x:.{nd}f}"


def main():
    data = {p.stem: json.loads(p.read_text()) for p in sorted(RES.glob("*.json"))}
    for gpu, d in data.items():
        if "env" not in d:
            print(f"\n=== {gpu}: {d.get('error', 'no results')}"); continue
        print(f"\n=== {gpu}  ({d['env'].get('gpu')}, {d['env'].get('model')}, vLLM {d['env'].get('vllm')})")
        s = d.get("single", {}).get("result", {})
        if "ttft_ms" in s:
            print(f"single request: TTFT p50 {f(s['ttft_ms']['p50'])} ms, TPOT p50 {f(s['tpot_ms']['p50'], 2)} ms, "
                  f"ITL p50 {f(s['itl_ms']['p50'], 2)} ms, text chunks per token {s.get('text_chunks_per_token', 0):.2f}")
            if s.get("tpot_ms", {}).get("p50") and s["itl_ms"]["p50"] and s["tpot_ms"]["p50"] < 0.8 * s["itl_ms"]["p50"]:
                print("  WARNING: TPOT p50 is well below ITL p50; the end of the stream was probably not timed")
        print("rate  req/s  out tok/s  TTFT p50/p99 (ms)  TPOT p50/p99 (ms)  ITL p99  goodput  SLO met")
        for r in d.get("sweep", {}).get("runs", []):
            if "ttft_ms" not in r:
                print("error", r.get("error", "")[:200]); continue
            print(f"{r['rate']:>4}  {r['request_throughput']:5.2f}  {r['output_token_throughput']:9.0f}  "
                  f"{f(r['ttft_ms']['p50']):>6}/{f(r['ttft_ms']['p99']):<7}  {f(r['tpot_ms']['p50'], 1):>6}/"
                  f"{f(r['tpot_ms']['p99'], 1):<7}  {f(r['itl_ms']['p99'], 1):>6}  {r['goodput_req_s']:6.2f}  "
                  f"{100 * r['slo_attainment']:5.1f}%")
        runs = d.get("long", {}).get("runs", [])
        if runs:
            print("long prompts (4K): rate  out tok/s  TTFT p50/p99 (ms)  TPOT p50 (ms)  goodput  peak KV use  preemptions")
        for r in runs:
            if "ttft_ms" not in r:
                print("error", r.get("error", "")[:200]); continue
            after, peak = r.get("metrics_after", {}), r.get("metrics_peak", {})
            kv = max([v for k, v in peak.items() if "cache_usage" in k] or [None])
            pre = sum(v for k, v in after.items() if "preemption" in k) - sum(
                v for k, v in r.get("metrics_before", {}).items() if "preemption" in k)
            print(f"  {r['rate']:>5}  {r['output_token_throughput']:9.0f}  {f(r['ttft_ms']['p50']):>6}/{f(r['ttft_ms']['p99']):<7}"
                  f"  {f(r['tpot_ms']['p50'], 1):>6}  {r['goodput_req_s']:6.2f}  {f(kv, 2):>6}  {pre:6.0f}")
        for name in ("prefix_on", "prefix_off"):
            p = d.get("prefix", {}).get(name, {}).get("follow_up", {})
            if "ttft_ms" in p:
                print(f"{name}: follow-up TTFT p50 {f(p['ttft_ms']['p50'])} ms, p99 {f(p['ttft_ms']['p99'])} ms")
        for name, b in d.get("budget", {}).items():
            r = b.get("result", {})
            if "ttft_ms" in r:
                print(f"{name} @ {b['rate']} req/s: TTFT p99 {f(r['ttft_ms']['p99'])} ms, ITL p99 {f(r['itl_ms']['p99'], 1)} ms, "
                      f"TPOT p50 {f(r['tpot_ms']['p50'], 1)} ms")
        for name, sp in d.get("spec", {}).items():
            if not isinstance(sp, dict) or "runs" not in sp:
                continue
            for rate, r in sp["runs"].items():
                if "tpot_ms" in r:
                    print(f"{name} @ {rate} req/s: TPOT p50 {f(r['tpot_ms']['p50'], 2)} ms, "
                          f"out tok/s {r['output_token_throughput']:.0f}")
            m = sp.get("metrics", {})
            # Prometheus keys carry labels, e.g. vllm:spec_decode_num_draft_tokens_total{model_name="..."}
            base = lambda k: k.split("{")[0]
            acc = sum(v for k, v in m.items() if base(k) == "vllm:spec_decode_num_accepted_tokens_total")
            dra = sum(v for k, v in m.items() if base(k) == "vllm:spec_decode_num_draft_tokens_total")
            if dra:
                print(f"{name}: draft acceptance {acc / dra:.2f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
        for gpu, d in data.items():
            runs = [r for r in d.get("sweep", {}).get("runs", []) if "ttft_ms" in r]
            if not runs:
                continue
            x = [r["rate"] for r in runs]
            ax[0].plot(x, [r["output_token_throughput"] for r in runs], marker="o", label=gpu)
            ax[1].plot(x, [r["ttft_ms"]["p99"] for r in runs], marker="o", label=gpu)
            ax[2].plot(x, [r["goodput_req_s"] for r in runs], marker="o", label=gpu)
            # gpu here is the results file stem, e.g. A100-80GB_gemma
        for a, t in zip(ax, ("output tokens/s", "TTFT p99 (ms)", "goodput (req/s)")):
            a.set_xlabel("offered load (req/s)"); a.set_ylabel(t); a.legend()
        ax[1].set_yscale("log")
        fig.tight_layout(); fig.savefig(RES / "sweep.png", dpi=200)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
