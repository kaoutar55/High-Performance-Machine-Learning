"""GPTQ on one layer: the same 4 or 3 bits as round-to-nearest, with error-compensated rounding.

Round-to-nearest (RTN) quantizes every weight independently. GPTQ (Frantar et al., ICLR 2023) quantizes the
columns of W one at a time and, after each column, updates the columns not yet quantized to cancel the error it
just made on the layer's output for the calibration activations X. The update uses the inverse of H = 2 X^T X.
Both use the same symmetric per-output-channel scales, so the only difference is how each weight is rounded.
Uses the weights and outlier channels of the layer in quant_basics.py, with correlated activations.

    python gptq_demo.py                    # CPU, about a minute
"""
import torch

from quant_basics import layer, rel_err


def quant_rows(w, scale, qmax):
    """Symmetric quantize-dequantize of a column (one value per output row) with per-row scales."""
    return torch.clamp(torch.round(w / scale), -qmax, qmax) * scale


def rtn(W, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = W.abs().amax(1).clamp(min=1e-12) / qmax
    return torch.clamp(torch.round(W / scale[:, None]), -qmax, qmax) * scale[:, None]


def gptq(W, X, bits, damp=0.01, block=128):
    """GPTQ with per-row scales, lazy block updates and a Cholesky factor of H^-1 (the paper's Algorithm 1)."""
    W = W.clone().double()
    n_in = W.shape[1]
    qmax = 2 ** (bits - 1) - 1
    scale = W.abs().amax(1).clamp(min=1e-12) / qmax
    H = 2 * X.double().T @ X.double()
    H += damp * H.diagonal().mean() * torch.eye(n_in, dtype=H.dtype)   # dampening for numerical stability
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    for b0 in range(0, n_in, block):
        b1 = min(b0 + block, n_in)
        Wb, Eb = W[:, b0:b1].clone(), torch.zeros(W.shape[0], b1 - b0, dtype=W.dtype)
        for i in range(b1 - b0):
            w, d = Wb[:, i], Hinv[b0 + i, b0 + i]
            q = quant_rows(w, scale, qmax)
            Q[:, b0 + i] = q
            err = (w - q) / d
            Wb[:, i:] -= err[:, None] * Hinv[b0 + i, b0 + i:b1][None, :]     # compensate inside the block
            Eb[:, i] = err
        W[:, b1:] -= Eb @ Hinv[b0:b1, b1:]                                   # lazy update of later blocks
    return Q.float()


if __name__ == "__main__":
    W, _, outliers = layer()
    d = W.shape[1]
    # LLM activations are correlated across channels. Independent channels would make H diagonal, and then
    # GPTQ has nothing to compensate with and matches RTN. Here 1,024 shared factors plus noise give each row
    # correlated channels; the size of GPTQ's gain depends on how strong that correlation is.
    X = torch.randn(9216, 1024) @ (torch.randn(1024, d) / 32) + 0.3 * torch.randn(9216, d)
    X[:, outliers] *= 50.0                        # the same outlier channels as the layer in quant_basics.py
    calib, test = X[:8192], X[8192:]              # fit on 8,192 rows, measure on 1,024 held-out rows
    for bits in (4, 3):
        e_rtn = rel_err(test, W, test, rtn(W, bits))
        e_gptq = rel_err(test, W, test, gptq(W, calib, bits))
        print(f"INT{bits} per-channel  RTN {e_rtn:.4f}  GPTQ {e_gptq:.4f}  ratio {e_rtn / e_gptq:.2f}x")
