"""Pruning one layer to 50% sparsity: magnitude vs Wanda scores, unstructured vs the 2:4 pattern.

Magnitude keeps the largest |W|. Wanda (Sun et al., ICLR 2024) keeps the largest |W_ij| * ||X_j||, so a small
weight that multiplies a large activation channel survives. 2:4 keeps 2 of every 4 consecutive weights in a
row, the pattern sparse Tensor Cores accelerate. Uses the layer from quant_basics.py.

    python prune24.py
"""
import torch

from quant_basics import layer, rel_err


def mask_unstructured(score, sparsity=0.5):
    """Per output row, drop the lowest-scoring fraction of weights (Wanda compares weights within a row)."""
    k = int(score.shape[1] * sparsity)
    drop = score.argsort(dim=1)[:, :k]
    return torch.ones_like(score).scatter_(1, drop, 0.0)


def mask_24(score):
    """In every group of 4 consecutive weights of a row, keep the 2 highest-scoring ones."""
    g = score.reshape(-1, 4)
    drop = g.argsort(dim=1)[:, :2]
    return torch.ones_like(g).scatter_(1, drop, 0.0).reshape(score.shape)


if __name__ == "__main__":
    W, X, _ = layer()
    scores = {"magnitude": W.abs(), "Wanda": W.abs() * X.norm(dim=0)}
    for name, score in scores.items():
        for pattern, mask in (("unstructured 50%", mask_unstructured(score)), ("2:4", mask_24(score))):
            print(f"{name:9s} {pattern:16s} output error {rel_err(X, W, X, W * mask):.4f}")
