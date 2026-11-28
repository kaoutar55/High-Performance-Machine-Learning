"""Chapter 10 vs Chapter 8 capacity ablation: Llama-3.1-8B-Instruct BF16 on one H100, vLLM 0.33.0.

Chapter 8 measured 48 req/s and Chapter 10 64 req/s for the same model, GPU and objectives (TTFT 500 ms,
TPOT 50 ms). This driver runs a 2 x 2 design with one load generator for all cells:

    A  Chapter 8 prompts (prompts_single.jsonl, 1,000 RAG prompts)   prefix caching off
    B  Chapter 8 prompts                                             prefix caching on
    C  Chapter 10 prompts (HotpotQA eval.jsonl[:500], chat template) prefix caching off
    D  Chapter 10 prompts                                            prefix caching on

Every request goes through chapter08/loadgen.py's one_request() (completions endpoint, ignore_eos, 128 tokens,
special tokens streamed) on loadgen.run()'s open-loop Poisson schedule, and every per-request record is kept, so
both capacity rules are computed from the same records:

    Chapter 8 rule   share of sent requests with TTFT <= 500 ms and TPOT <= 50 ms, TPOT = (e2e - TTFT) / (tokens - 1);
                     capacity = highest rate at which every repeat reaches >= 90%
    Chapter 10 rule  serve_bench.sweep(): the same share with TPOT = (end - first chunk) / (chunks - 1), where the
                     chat stream adds a role chunk and a finish chunk; capacity = highest rate with >= 90% in one run

Server: vllm serve MODEL --dtype bfloat16 --max-model-len 4096, prefix caching on or off. Repeat 1 of each cell
sweeps every rate in ascending order; repeats 2 and 3 run the rates around repeat 1's knee in a rotated order.
Each repeat starts a fresh server and warms it up for 60 s at the lowest rate (as Chapter 8's knee study).

    python capacity_ablation.py prep --ch8 prompts_single.jsonl --hotpot eval.jsonl --out DIR
    python capacity_ablation.py cell --cell A --prompts DIR/ch8.jsonl --out raw/A.json [--deadline EPOCH]
"""
import argparse
import asyncio
import json
import os
import random
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "chapter08"))
import loadgen  # noqa: E402  (Chapter 8's request function and percentile helper)

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
RATES = [32, 40, 48, 56, 64, 72, 80]
SLO_TTFT_MS, SLO_TPOT_MS, MAX_TOKENS = 500, 50, 128
SECONDS_PER_RATE, WARMUP_S, REPEATS = 60, 60, 3
CELLS = {"A": ("ch8", False), "B": ("ch8", True), "C": ("hotpot", False), "D": ("hotpot", True),
         # optional: D with serve_bench.sweep()'s prompt choice (random with replacement, one seed per rate)
         "D10": ("hotpot", True)}
GPU_QUERY = "clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active"


# ------------------------------------------------------------------------------------------------ prompts
def prep(ch8, hotpot, out_dir):
    """Write ch8.jsonl (unchanged) and hotpot.jsonl (eval.jsonl[:500] rendered with the Llama chat template, as
    vLLM's chat endpoint does; the template's leading <|begin_of_text|> is removed because the completions endpoint
    adds BOS itself), and the prompt-length distributions of both."""
    from transformers import AutoTokenizer
    sys.path.insert(0, HERE)
    from common import for_model
    tok = AutoTokenizer.from_pretrained(MODEL)
    os.makedirs(out_dir, exist_ok=True)
    rows8 = [json.loads(l) for l in open(ch8)]
    with open(os.path.join(out_dir, "ch8.jsonl"), "w") as f:
        for r in rows8:
            f.write(json.dumps({"prompt": r["prompt"]}) + "\n")
    rowsh = [json.loads(l) for l in open(hotpot)][:500]
    bos = tok.bos_token or ""
    with open(os.path.join(out_dir, "hotpot.jsonl"), "w") as f:
        for r in rowsh:
            text = tok.apply_chat_template(for_model("llama-3.1-8b", r["messages"]), tokenize=False,
                                           add_generation_prompt=True)
            if bos and text.startswith(bos):
                text = text[len(bos):]
            f.write(json.dumps({"prompt": text}) + "\n")
    stats = {}
    for name in ("ch8", "hotpot"):
        lens = sorted(len(tok(json.loads(l)["prompt"], add_special_tokens=True)["input_ids"])
                      for l in open(os.path.join(out_dir, f"{name}.jsonl")))
        stats[name] = {"prompts": len(lens), "tokens_median": loadgen.pct(lens, 50), "tokens_p90": loadgen.pct(lens, 90),
                       "tokens_min": lens[0], "tokens_max": lens[-1], "tokens_mean": statistics.fmean(lens),
                       "output_tokens": f"{MAX_TOKENS} for every request (ignore_eos), as in both chapters"}
    # the same HotpotQA prompts tokenized by the chat template directly, to confirm the rendering
    direct = sorted(len(tok.apply_chat_template(for_model("llama-3.1-8b", r["messages"]), tokenize=True,
                                                add_generation_prompt=True)) for r in rowsh)
    stats["hotpot"]["chat_template_tokens_median"] = loadgen.pct(direct, 50)
    json.dump(stats, open(os.path.join(out_dir, "prompt_stats.json"), "w"), indent=2)
    print(json.dumps(stats), flush=True)


# ------------------------------------------------------------------------------------------------ server
class Server:
    def __init__(self, prefix_caching):
        with socket.socket() as s:
            s.bind(("", 0))
            self.port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.cmd = ["vllm", "serve", MODEL, "--dtype", "bfloat16", "--max-model-len", "4096", "--port", str(self.port),
                    "--enable-prefix-caching" if prefix_caching else "--no-enable-prefix-caching"]

    def __enter__(self):
        self.t0 = time.time()
        self.log = open(f"/tmp/{os.environ.get('USER', 'u')}_vllm_{self.port}.log", "w")
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(1200):
            try:
                if urllib.request.urlopen(f"{self.url}/health", timeout=2).status == 200:
                    self.startup_s = time.time() - self.t0
                    return self
            except Exception:  # noqa: BLE001
                pass
            if self.proc.poll() is not None:
                break
            time.sleep(1)
        self.log.flush()
        raise RuntimeError("vllm did not start:\n" + open(self.log.name).read()[-3000:])

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()

    def prefix_counters(self):
        """vLLM prefix-cache counters (tokens): queries and hits, from /metrics."""
        out = {}
        try:
            for line in urllib.request.urlopen(f"{self.url}/metrics", timeout=5).read().decode().splitlines():
                if line.startswith("vllm:prefix_cache") and not line.startswith("#"):
                    name, val = line.rsplit(" ", 1)
                    key = name.split("{")[0]
                    out[key] = out.get(key, 0.0) + float(val)
        except Exception as e:  # noqa: BLE001
            out["error"] = str(e)[:200]
        return out


class GpuLog:
    """nvidia-smi once per second during a run; summary of SM clock, power and clock-event reasons."""

    def __enter__(self):
        self.path = f"/tmp/{os.environ.get('USER', 'u')}_gpu_{os.getpid()}.csv"
        self.f = open(self.path, "w")
        self.p = subprocess.Popen(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits",
                                   "-lms", "1000"], stdout=self.f, stderr=subprocess.STDOUT)
        return self

    def __exit__(self, *exc):
        self.p.terminate()
        self.p.wait(10)
        self.f.close()

    def summary(self):
        rows = []
        for line in open(self.path):
            p = [x.strip() for x in line.split(",")]
            try:
                rows.append((float(p[0]), float(p[2]), p[4]))
            except (ValueError, IndexError):
                pass
        if not rows:
            return {}
        sm = sorted(r[0] for r in rows)
        return {"samples": len(rows), "sm_clock_mhz_start": rows[0][0], "sm_clock_mhz_end": rows[-1][0],
                "sm_clock_mhz_p50": sm[len(sm) // 2], "sm_clock_mhz_min": sm[0], "power_w_start": rows[0][1],
                "power_w_end": rows[-1][1], "power_w_max": max(r[1] for r in rows),
                "clock_event_reasons": sorted({r[2] for r in rows if r[2] not in ("0x0000000000000000", "0x0")})}


# ------------------------------------------------------------------------------------------------ load
async def load(url, prompts, rate, num, offset, seed, seen, random_order=False):
    """loadgen.run()'s schedule (Poisson arrivals on an absolute clock, prompts in order from `offset`), keeping
    every per-request record. `seen` holds the prompt indices already sent to this server (prompt reuse)."""
    import aiohttp
    order = list(range(offset, len(prompts))) + list(range(offset))
    rng = random.Random(seed)
    pick = random.Random(seed + 1)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    recs, tasks = [], []
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        start = time.perf_counter()
        t_next, lags = 0.0, []
        for i in range(num):
            if rate > 0 and i > 0:
                t_next += rng.expovariate(rate)
                delay = start + t_next - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            lags.append(time.perf_counter() - start - t_next)
            idx = pick.randrange(len(prompts)) if random_order else order[i % len(order)]
            recs.append({"prompt_idx": idx, "reused": idx in seen, "t_sched": t_next})
            seen.add(idx)
            tasks.append(asyncio.create_task(loadgen.one_request(session, url, MODEL, prompts[idx], MAX_TOKENS)))
            await asyncio.sleep(0)
        for rec, t in zip(recs, tasks):
            r = await t
            if r and not r.get("failed"):
                rec.update({k: r[k] for k in ("ttft", "e2e", "tpot", "out_tokens", "text_chunks")})
            else:
                rec["failed"] = True
        duration = time.perf_counter() - start
    return recs, duration, max(lags) if lags else 0.0


def rules(recs, duration):
    """Both capacity rules' per-run share, and latency percentiles, from the per-request records."""
    n = len(recs)
    ok = [r for r in recs if not r.get("failed")]
    ch8 = sum(1 for r in ok if r["ttft"] * 1e3 <= SLO_TTFT_MS and (r["tpot"] or 0) * 1e3 <= SLO_TPOT_MS) / max(n, 1)

    def tpot10(r):                       # serve_bench.one(): (end - first) / (n_lines - 1); chat adds role + finish
        lines = r["text_chunks"] + 2      # chat stream: role chunk (with the first token) + one per token + finish
        return (r["e2e"] - r["ttft"]) / max(lines - 1, 1)
    ch10 = sum(1 for r in ok if r["ttft"] <= SLO_TTFT_MS / 1e3 and tpot10(r) <= SLO_TPOT_MS / 1e3) / max(n, 1)
    ttft = [r["ttft"] * 1e3 for r in ok]
    tpot = [r["tpot"] * 1e3 for r in ok if r["tpot"] is not None]
    return {"requests": n, "failed": n - len(ok), "duration_s": duration, "request_throughput": len(ok) / duration,
            "share_ch8_rule": ch8, "share_ch10_rule": ch10,
            "ttft_ms": {f"p{q}": loadgen.pct(ttft, q) for q in (50, 90, 99)},
            "tpot_ms": {f"p{q}": loadgen.pct(tpot, q) for q in (50, 99)},
            "reused_prompt_share": sum(r["reused"] for r in recs) / max(n, 1),
            "text_chunks_per_token": sum(r.get("text_chunks", 0) for r in ok) / max(1, sum(r.get("out_tokens", 0) for r in ok))}


def knee_rates(rep1):
    passing = [r for r in RATES if rep1.get(str(r), {}).get("share_ch8_rule", 0) >= 0.9]
    top = max(passing) if passing else RATES[0]
    i = RATES.index(top)
    sel = RATES[max(0, i - 1):i + 2]
    while len(sel) < 3:                                    # at the ends of the grid keep three rates
        sel = RATES[-3:] if i >= len(RATES) - 1 else RATES[:3]
    return sel


def long_check(prompts_path, cell_a_json, out, deadline=None):
    """Cell A at its repeat-1 knee rate for 120 s of arrivals (fresh server, same warm-up), against the 60 s runs."""
    a = json.load(open(cell_a_json))
    rep1 = a["repeats"][0]["runs"]
    rate = max([r for r in RATES if rep1.get(str(r), {}).get("share_ch8_rule", 0) >= 0.9], default=RATES[0])
    if deadline and time.time() + 120 + WARMUP_S + 150 > deadline:
        json.dump({"skipped": "budget", "rate": rate}, open(out, "w"))
        return
    prompts = [json.loads(l)["prompt"] for l in open(prompts_path)]
    with Server(False) as s:
        seen = set()
        asyncio.run(load(s.url, prompts, min(RATES), min(RATES) * WARMUP_S, offset=850, seed=950, seen=seen))
        with GpuLog() as g:
            recs, dur, lag = asyncio.run(load(s.url, prompts, rate, rate * 120, offset=17, seed=990, seen=seen))
        r = rules(recs, dur)
        r.update({"cell": "A", "rate": rate, "seconds_of_arrivals": 120, "gpu": g.summary(),
                  "share_ch8_rule_60s_runs": [x["runs"][str(rate)]["share_ch8_rule"] for x in a["repeats"]
                                              if "runs" in x and str(rate) in x["runs"]]})
    json.dump(r, open(out, "w"), indent=1)
    print(json.dumps({"check_120s_rate": rate, "ch8": round(r["share_ch8_rule"], 3)}), flush=True)


def cell(name, prompts_path, out, deadline=None, long_check=False):
    kind, pc = CELLS[name]
    rnd = name == "D10"
    prompts = [json.loads(l)["prompt"] for l in open(prompts_path)]
    res = {"cell": name, "prompts": kind, "prefix_caching": pc, "prompt_order": "random" if rnd else "cyclic", "rates": RATES, "seconds_per_rate": SECONDS_PER_RATE,
           "repeats": []}

    def save():
        json.dump(res, open(out, "w"), indent=1)

    rep1 = {}
    for rep in range(1 if rnd else REPEATS):
        rates = RATES if rep == 0 else (lambda k: k[rep:] + k[:rep])(knee_rates(rep1))
        need = 120 + WARMUP_S + len(rates) * (SECONDS_PER_RATE + 15)
        if deadline and time.time() + need > deadline:
            res["repeats"].append({"repeat": rep + 1, "skipped": f"budget: needs about {need} s"})
            save()
            break
        with Server(pc) as s:
            seen = set()
            warm = min(RATES)
            w, _, _ = asyncio.run(load(s.url, prompts, warm, warm * WARMUP_S, offset=(850 if kind == "ch8" else 450) % len(prompts),
                                       seed=900 + rep, seen=seen))
            runs = {}
            for i, rate in enumerate(rates):
                c0 = s.prefix_counters()
                with GpuLog() as g:
                    recs, dur, lag = asyncio.run(load(s.url, prompts, rate, rate * SECONDS_PER_RATE,
                                                      offset=(rep * 211 + i * 131) % len(prompts), seed=200 + 10 * rep + i,
                                                      seen=seen, random_order=rnd))
                c1 = s.prefix_counters()
                r = rules(recs, dur)
                q = next((k for k in c1 if k.endswith("queries_total") or k.endswith("queries")), None)
                h = next((k for k in c1 if k.endswith("hits_total") or k.endswith("hits")), None)
                dq = c1.get(q, 0) - c0.get(q, 0) if q else 0
                r["prefix_hit_rate_tokens"] = (c1.get(h, 0) - c0.get(h, 0)) / dq if dq else None
                r.update({"rate": rate, "position_in_repeat": i, "send_lag_max_ms": 1e3 * lag, "gpu": g.summary()})
                r["records"] = [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in x.items()} for x in recs]
                runs[str(rate)] = r
                print(json.dumps({"cell": name, "repeat": rep + 1, "rate": rate, "ch8": round(r["share_ch8_rule"], 3),
                                  "ch10": round(r["share_ch10_rule"], 3), "ttft_p90": round(r["ttft_ms"]["p90"] or 0, 1),
                                  "hit": r["prefix_hit_rate_tokens"], "reused": round(r["reused_prompt_share"], 3)}), flush=True)
            res["repeats"].append({"repeat": rep + 1, "order": rates, "startup_s": s.startup_s,
                                   "warmup": {"rate": warm, "requests": len(w)}, "runs": runs})
            if rep == 0:
                rep1 = runs
        save()
    if long_check and name == "A" and (not deadline or time.time() + 400 < deadline):
        rate = max([r for r in RATES if rep1.get(str(r), {}).get("share_ch8_rule", 0) >= 0.9], default=RATES[0])
        with Server(pc) as s:
            seen = set()
            asyncio.run(load(s.url, prompts, min(RATES), min(RATES) * WARMUP_S, offset=850, seed=950, seen=seen))
            recs, dur, lag = asyncio.run(load(s.url, prompts, rate, rate * 120, offset=17, seed=990, seen=seen))
            r = rules(recs, dur)
            r.update({"rate": rate, "seconds": 120})
            res["long_check_120s"] = r
            print(json.dumps({"cell": name, "check_120s_rate": rate, "ch8": round(r["share_ch8_rule"], 3)}), flush=True)
    # capacities
    reps = [x for x in res["repeats"] if "runs" in x]

    def cap(rule, all_repeats):
        good = []
        for rate in RATES:
            shares = [x["runs"][str(rate)][rule] for x in reps if str(rate) in x["runs"]]
            if not shares:
                continue
            if (all(s_ >= 0.9 for s_ in shares) and (len(shares) == len(reps) if all_repeats else True)):
                good.append(rate)
        return max(good) if good else None
    res["capacity"] = {"ch8_rule_all_repeats": cap("share_ch8_rule", True),
                       "ch10_rule_repeat1": max([r for r in RATES if reps and reps[0]["runs"].get(str(r), {}).get("share_ch10_rule", 0) >= 0.9], default=None) if reps else None,
                       "ch8_rule_repeat1": max([r for r in RATES if reps and reps[0]["runs"].get(str(r), {}).get("share_ch8_rule", 0) >= 0.9], default=None) if reps else None}
    save()
    print(json.dumps({"cell": name, "capacity": res["capacity"]}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["prep", "cell", "check120"])
    ap.add_argument("--cell-a-json")
    ap.add_argument("--ch8")
    ap.add_argument("--hotpot")
    ap.add_argument("--cell", choices=list(CELLS))
    ap.add_argument("--prompts")
    ap.add_argument("--deadline", type=float)
    ap.add_argument("--long-check", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.what == "prep":
        prep(a.ch8, a.hotpot, a.out)
    elif a.what == "check120":
        long_check(a.prompts, a.cell_a_json, a.out, a.deadline)
    else:
        cell(a.cell, a.prompts, a.out, a.deadline, a.long_check)
