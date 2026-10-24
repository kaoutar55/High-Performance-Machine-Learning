"""Megatron-style tensor parallelism for a transformer MLP, simulated on one device.

The first weight A is split by columns and the second weight B by rows. Each "rank" computes
gelu(x @ A_i) @ B_i, and one all-reduce (here a sum) produces the full output. GELU is elementwise, so splitting
A by columns needs no communication before it.
"""
import torch
import torch.nn.functional as F


def single_gpu_mlp(x: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return F.gelu(x @ A) @ B


def tensor_parallel_mlp(x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, tp: int = 2):
    """Returns (output, number of forward all-reduces). A is [d, h], B is [h, d]; h must divide by tp."""
    h = A.shape[1]
    if h % tp:
        raise ValueError(f"hidden size {h} is not divisible by tp={tp}")
    A_parts = torch.chunk(A, tp, dim=1)   # column split: each rank holds h/tp output features
    B_parts = torch.chunk(B, tp, dim=0)   # row split: each rank holds the matching h/tp input rows
    partials = [F.gelu(x @ Ai) @ Bi for Ai, Bi in zip(A_parts, B_parts)]
    out = torch.stack(partials).sum(dim=0)  # the single all-reduce of the forward pass
    return out, 1


if __name__ == "__main__":
    torch.manual_seed(0)
    x, A, B = torch.randn(8, 16, dtype=torch.float64), torch.randn(16, 64, dtype=torch.float64), torch.randn(64, 16, dtype=torch.float64)
    out, n = tensor_parallel_mlp(x, A, B, 4)
    print((out - single_gpu_mlp(x, A, B)).abs().max().item(), n)
