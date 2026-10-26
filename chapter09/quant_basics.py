"""Linear quantization from first principles: scale, zero point, granularity, and the error each choice leaves.

A 4096x4096 weight matrix and 512 activation rows are built to have the two properties of real LLM layers that
matter for quantization: output channels with very different ranges, and a few input channels whose activations
are 50x larger than the rest. The script fake-quantizes them (quantize, then dequantize back to float) and
reports the relative error of the layer's output, ||XW^T - X'W'^T|| / ||XW^T||.

    python quant_basics.py                 # CPU is fine
"""
import torch

torch.manual_seed(0)


def fake_quant(x, bits=8, granularity="tensor", group=128, symmetric=True):
    """Quantize x to integers and map back. granularity: one scale per tensor, per row (channel), per group of
    `group` consecutive values in a row, or per token (row of an activation matrix, same as 'channel')."""
    shape = x.shape
    if granularity == "tensor":
        v = x.reshape(1, -1)
    elif granularity in ("channel", "token"):
        v = x.reshape(x.shape[0], -1)
    else:                                                   # "group"
        v = x.reshape(-1, group)
    if symmetric:                                           # zero point 0, range [-2^(b-1), 2^(b-1)-1]
        qmax = 2 ** (bits - 1) - 1
        scale = v.abs().amax(1, keepdim=True).clamp(min=1e-12) / qmax
        q = torch.clamp(torch.round(v / scale), -qmax - 1, qmax)
        out = q * scale
    else:                                                   # zero point maps the real 0 to an integer
        qmax = 2 ** bits - 1
        lo, hi = v.amin(1, keepdim=True), v.amax(1, keepdim=True)
        scale = (hi - lo).clamp(min=1e-12) / qmax
        zp = torch.round(-lo / scale)
        q = torch.clamp(torch.round(v / scale) + zp, 0, qmax)
        out = (q - zp) * scale
    return out.reshape(shape)


def fake_fp8(x, granularity="channel"):
    """FP8 E4M3 with a per-row (or per-tensor) scale that maps the largest value to 448."""
    v = x if granularity == "channel" else x.reshape(1, -1)
    scale = v.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
    return ((v / scale).to(torch.float8_e4m3fn).to(torch.float32) * scale).reshape(x.shape)


def layer():
    d = 4096
    W = torch.randn(d, d) * 0.02 * torch.exp(torch.randn(d, 1) * 0.5)    # rows with different ranges
    X = torch.randn(512, d)
    outliers = torch.randperm(d)[:8]
    X[:, outliers] *= 50.0                                                # a few large activation channels
    return W, X, outliers


def rel_err(X, W, Xq, Wq):
    Y = X @ W.T
    return ((Y - Xq @ Wq.T).norm() / Y.norm()).item()


if __name__ == "__main__":
    W, X, _ = layer()
    print("weights only (activations kept in FP32)")
    for name, Wq in [("INT8 per-tensor", fake_quant(W, 8, "tensor")),
                     ("INT8 per-channel", fake_quant(W, 8, "channel")),
                     ("FP8 per-channel", fake_fp8(W)),
                     ("INT4 per-channel", fake_quant(W, 4, "channel")),
                     ("INT4 group 128", fake_quant(W, 4, "group", 128)),
                     ("INT4 group 128 asym", fake_quant(W, 4, "group", 128, symmetric=False))]:
        print(f"  {name:22s} {rel_err(X, W, X, Wq):.4f}")
    Wq = fake_quant(W, 8, "channel")
    print("weights INT8 per-channel, plus activations")
    for name, Xq in [("INT8 per-tensor", fake_quant(X, 8, "tensor")),
                     ("INT8 per-token", fake_quant(X, 8, "token")),
                     ("FP8 per-token", fake_fp8(X))]:
        print(f"  {name:22s} {rel_err(X, W, Xq, Wq):.4f}")
