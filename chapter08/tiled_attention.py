"""Attention for one head computed tile by tile with an online softmax, and the split-KV combine of
Flash-Decoding. Checks both against the reference softmax(QK^T)V.

    python tiled_attention.py
"""
import torch


def tiled_attention(q, k, v, tile=128):          # q: (N, d), k and v: (M, d)
    n, d = q.shape
    scale = d ** -0.5
    o = torch.zeros_like(q)
    m = torch.full((n, 1), float("-inf"), dtype=q.dtype, device=q.device)
    l = torch.zeros((n, 1), dtype=q.dtype, device=q.device)
    for s in range(0, k.shape[0], tile):
        kt, vt = k[s:s + tile], v[s:s + tile]
        scores = (q @ kt.T) * scale
        m_new = torch.maximum(m, scores.amax(dim=-1, keepdim=True))
        p = torch.exp(scores - m_new)
        corr = torch.exp(m - m_new)
        l = l * corr + p.sum(dim=-1, keepdim=True)
        o = o * corr + p @ vt
        m = m_new
    return o / l


def split_kv_attention(q, k, v, splits=4):
    """Flash-Decoding: independent partial results per key chunk, combined with their log-sum-exp."""
    scale = q.shape[-1] ** -0.5
    outs, lses = [], []
    for kc, vc in zip(k.chunk(splits), v.chunk(splits)):
        s = (q @ kc.T) * scale
        lse = torch.logsumexp(s, dim=-1, keepdim=True)
        outs.append(torch.exp(s - lse) @ vc)
        lses.append(lse)
    w = torch.softmax(torch.cat(lses, dim=-1), dim=-1)          # weight of each chunk
    return sum(w[:, i:i + 1] * outs[i] for i in range(splits))


if __name__ == "__main__":
    torch.manual_seed(0)
    q, k, v = (torch.randn(1024, 64, dtype=torch.float64) for _ in range(3))
    ref = torch.softmax(q @ k.T / 8, dim=-1) @ v
    print("tiled   max error", (tiled_attention(q, k, v) - ref).abs().max().item())
    print("split-KV max error", (split_kv_attention(q[:1], k, v) - ref[:1]).abs().max().item())
