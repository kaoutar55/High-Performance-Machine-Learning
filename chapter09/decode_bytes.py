"""Bytes read per decode step and the batch-1 latency floor for each weight format.

At batch 1, a decode step reads every weight once, so its time is at least (weight bytes) / (memory bandwidth).
Embeddings and the output projection (lm_head) usually stay in 16 bits; group-128 INT4 adds one FP16 scale per
128 weights. Gemma-2B ties its 256K-token output projection to the embedding, so that 16-bit matrix is a large
share of its bytes once the linear layers are compressed.

    python decode_bytes.py
"""
MODELS = {   # parameters in quantizable linear layers, and in the 16-bit lm_head that every step reads
    "Gemma-2B": {"linear": 1.982e9, "head": 0.524e9},
    "Llama-3.1-8B": {"linear": 6.979e9, "head": 0.525e9},
}
FORMATS = {"16-bit": 2.0, "8-bit": 1.0, "INT4 g128": 0.5 + 2 / 128}       # bytes per linear-layer weight
BANDWIDTH = {"T4": 320e9, "A100 80GB": 2.039e12, "H100 SXM": 3.35e12}   # bytes/s, datasheet


def step_bytes(m, fmt):
    return m["linear"] * FORMATS[fmt] + m["head"] * 2.0


if __name__ == "__main__":
    for name, m in MODELS.items():
        base = step_bytes(m, "16-bit")
        print(name)
        for fmt in FORMATS:
            b = step_bytes(m, fmt)
            floors = "  ".join(f"{g} {b / bw * 1e3:5.1f} ms" for g, bw in BANDWIDTH.items())
            print(f"  {fmt:9s} {b / 1e9:5.2f} GB per step  bound {base / b:4.2f}x   {floors}")
