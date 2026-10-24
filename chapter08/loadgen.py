"""Open-loop load generator for an OpenAI-compatible completions server (vLLM, SGLang, TensorRT-LLM).

Requests arrive as a Poisson process at --rate requests/s (open loop: arrivals do not wait for earlier
requests to finish, so queueing delay is measured instead of hidden). Each request streams its tokens; the
client records the arrival time of every chunk and derives, per request:

  TTFT  time from sending the request to the first streamed token
  ITL   gaps between consecutive streamed tokens
  TPOT  (end-to-end latency - TTFT) / (output tokens - 1)
  E2E   time from sending the request to the last token

and, over the run: request and output-token throughput, percentiles, and goodput (requests per second that
met both the TTFT and the TPOT objective).

    python loadgen.py --url http://localhost:8000 --model google/gemma-2b-it \
        --prompts prompts.jsonl --rate 4 --num 200 --max-tokens 128 --slo-ttft-ms 500 --slo-tpot-ms 50
"""
import argparse
import asyncio
import json
import random
import statistics
import time

import aiohttp


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


async def one_request(session, url, model, prompt, max_tokens):
    try:
        return await _one_request(session, url, model, prompt, max_tokens)
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError):
        return {"failed": True}


async def _one_request(session, url, model, prompt, max_tokens):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
            "stream": True, "ignore_eos": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    stamps, n_tokens = [], 0
    async with session.post(f"{url}/v1/completions", json=body) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                n_tokens = chunk["usage"]["completion_tokens"]
            if chunk.get("choices") and chunk["choices"][0].get("text"):
                stamps.append(time.perf_counter())
    if not stamps:
        return None
    ttft = stamps[0] - t0
    e2e = stamps[-1] - t0
    n = n_tokens or len(stamps)
    return {"ttft": ttft, "e2e": e2e, "tpot": (e2e - ttft) / (n - 1) if n > 1 else None,
            "itl": [b - a for a, b in zip(stamps, stamps[1:])], "out_tokens": n}
    # note: with speculative decoding one streamed chunk can carry several tokens, so ITL is per chunk


async def run(args):
    prompts = [json.loads(l)["prompt"] for l in open(args.prompts)]
    prompts = prompts[args.offset:] + prompts[:args.offset]
    rng = random.Random(args.seed)
    results, tasks = [], []
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        start = time.perf_counter()
        for i in range(args.num):
            tasks.append(asyncio.create_task(
                one_request(session, args.url, args.model, prompts[i % len(prompts)], args.max_tokens)))
            if args.rate > 0:
                await asyncio.sleep(rng.expovariate(args.rate))     # Poisson arrivals
        failed = 0
        for t in tasks:
            r = await t
            if r and not r.get("failed"):
                results.append(r)
            else:
                failed += 1
        duration = time.perf_counter() - start
    out = summarize(results, duration, args)
    out["failed"] = failed
    return out


def summarize(rs, duration, args):
    ttft = [r["ttft"] * 1e3 for r in rs]
    tpot = [r["tpot"] * 1e3 for r in rs if r["tpot"] is not None]
    itl = [x * 1e3 for r in rs for x in r["itl"]]
    e2e = [r["e2e"] * 1e3 for r in rs]
    ok = [r for r in rs if r["ttft"] * 1e3 <= args.slo_ttft_ms and (r["tpot"] or 0) * 1e3 <= args.slo_tpot_ms]
    out = {"rate": args.rate, "completed": len(rs), "duration_s": duration,
           "request_throughput": len(rs) / duration,
           "output_token_throughput": sum(r["out_tokens"] for r in rs) / duration,
           "goodput_req_s": len(ok) / duration, "slo_attainment": len(ok) / max(1, len(rs)),
           "slo": {"ttft_ms": args.slo_ttft_ms, "tpot_ms": args.slo_tpot_ms}}
    for name, xs in (("ttft_ms", ttft), ("tpot_ms", tpot), ("itl_ms", itl), ("e2e_ms", e2e)):
        out[name] = {"mean": statistics.fmean(xs) if xs else None,
                     **{f"p{q}": pct(xs, q) for q in (50, 90, 95, 99)}}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="JSONL file with a 'prompt' field per line")
    ap.add_argument("--rate", type=float, default=1.0, help="requests/s (Poisson); 0 sends all at once")
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--slo-ttft-ms", type=float, default=500)
    ap.add_argument("--slo-tpot-ms", type=float, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0, help="start at this prompt (keeps runs on distinct prompts)")
    a = ap.parse_args()
    print(json.dumps(asyncio.run(run(a)), indent=2))
