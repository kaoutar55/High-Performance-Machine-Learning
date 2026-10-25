"""Roofline estimate of the prefill and decode GPUs a workload needs (the pool-sizing listing of Chapter 8).

    python pd_size.py
"""


def decode_step_s(batch, ctx, gpu, m):
    """Roofline step time: read weights and all KV, or compute."""
    return max((m["w"] + batch * ctx * m["kv"]) / gpu["bw"],
               batch * m["flop_tok"] / (0.6 * gpu["peak"]))

def size_pools(rate, prompt, out, tpot_s, gpu, m, prefill_eff=0.5):
    prefill_tok_s = prefill_eff * gpu["peak"] / m["flop_tok"]      # compute-bound
    n_prefill = rate * prompt / prefill_tok_s
    ctx = prompt + out / 2                                          # mean context
    kv_fit = int((0.92 * gpu["mem"] - m["w"]) // (ctx * m["kv"]))    # sequences that fit
    fits = [b for b in range(1, kv_fit + 1)
            if decode_step_s(b, ctx, gpu, m) <= tpot_s]
    if not fits:
        raise ValueError("no batch size meets the TPOT objective")
    batch = max(fits)
    step = decode_step_s(batch, ctx, gpu, m)
    n_decode = rate * out * step / batch
    return n_prefill, n_decode, batch, kv_fit

h100 = {"peak": 989e12, "bw": 3.35e12, "mem": 80e9}
llama8b = {"w": 16.06e9, "kv": 131072, "flop_tok": 2 * 8.03e9}
workloads = [("RAG", 4000, 200), ("chat", 500, 400),
             ("reasoning", 1000, 4000)]
for name, prompt, out in workloads:
    p, d, b, fit = size_pools(20, prompt, out, 0.05, h100, llama8b)
    print(f"{name:9s} prefill {p:4.1f} decode {d:5.1f} GPUs "
          f"batch {b:3d} (KV limit {fit})")
