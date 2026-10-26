"""Bytes read per decode step and the batch-1 latency floor for each weight format.

At batch 1, a decode step reads every weight once, so its time is at least (weight bytes) / (memory bandwidth).
Embeddings and the output projection (lm_head) usually stay in 16 bits; group-128 INT4 adds one FP16 scale per
128 weights. Gemma-2B ties its 256K-token output projection to the embedding, so that 16-bit matrix is a large
share of its bytes once the linear layers are compressed.

    python decode_bytes.py
"""
MODELS = {   # parameters in quantizable linear layers, and in the 16-bit lm_head that every step reads
    "Gemma": {"linear": 1.982e9, "head": 0.524e9},           # Gemma-2B
    "Llama": {"linear": 6.979e9, "head": 0.525e9},           # Llama-3.1-8B
}
FORMATS = {"16-bit": 2.0, "8-bit": 1.0, "INT4 g128": 0.5 + 2 / 128}       # bytes per linear-layer weight
BANDWIDTH = {"T4": 320e9, "A100": 2.039e12, "H100": 3.35e12}   # bytes/s, datasheet (A100 80GB SXM, H100 SXM)


def step_bytes(m, fmt):
    return m["linear"] * FORMATS[fmt] + m["head"] * 2.0


if __name__ == "__main__":
    print("model  format     GB/step  bound    T4  A100  H100 (ms)")
    for name, m in MODELS.items():
        base = step_bytes(m, "16-bit")
        for fmt in FORMATS:
            b = step_bytes(m, fmt)
            ms = "".join(f"{b / bw * 1e3:6.1f}" for bw in BANDWIDTH.values())
            print(f"{name:6s} {fmt:9s} {b / 1e9:7.2f} {base / b:6.2f}x{ms}")
