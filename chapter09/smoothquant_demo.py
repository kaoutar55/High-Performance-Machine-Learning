"""SmoothQuant on one layer: move the activation outliers into the weights before W8A8 quantization.

For each input channel j, s_j = max|X_j|^alpha / max|W_j|^(1 - alpha). The layer computes the same product
with X / s and W * s (the division folds into the previous LayerNorm or linear layer at no runtime cost), and
both factors are now easier to quantize. Uses the layer from quant_basics.py.

    python smoothquant_demo.py
"""
import torch

from quant_basics import fake_quant, layer, rel_err


def smooth(X, W, alpha=0.5):
    s = X.abs().amax(0).pow(alpha) / W.abs().amax(0).pow(1 - alpha)
    return X / s, W * s


if __name__ == "__main__":
    W, X, outliers = layer()
    print(f"activation range: outlier channels {X[:, outliers].abs().max():.1f}, "
          f"others {X[:, [i for i in range(X.shape[1]) if i not in set(outliers.tolist())]].abs().max():.1f}")
    for alpha in (None, 0.5, 0.8):
        Xs, Ws = (X, W) if alpha is None else smooth(X, W, alpha)
        e_tensor = rel_err(X, W, fake_quant(Xs, 8, "tensor"), fake_quant(Ws, 8, "channel"))
        e_token = rel_err(X, W, fake_quant(Xs, 8, "token"), fake_quant(Ws, 8, "channel"))
        label = "no smoothing" if alpha is None else f"alpha = {alpha}"
        print(f"W8A8 {label:13s} per-tensor A {e_tensor:.4f}   per-token A {e_token:.4f}")
