"""Mixture-of-experts routing: top-1 gating, per-expert load, and capacity-limited dropping (GShard / Switch style)."""
import math

import torch


def route_top1(tokens: torch.Tensor, gate_w: torch.Tensor):
    """Top-1 routing. Returns (expert index per token, gate probability of the chosen expert)."""
    probs = torch.softmax(tokens @ gate_w, dim=-1)
    gate, assignments = probs.max(dim=-1)
    return assignments, gate


def counts_per_expert(assignments: torch.Tensor, num_experts: int) -> torch.Tensor:
    return torch.bincount(assignments, minlength=num_experts)


def expert_capacity(num_tokens: int, num_experts: int, capacity_factor: float) -> int:
    return math.ceil(capacity_factor * num_tokens / num_experts)


def apply_capacity(assignments: torch.Tensor, num_experts: int, capacity_factor: float):
    """Keep the first `capacity` tokens routed to each expert, in token order, and drop the rest.

    Returns (kept mask, number of dropped tokens, capacity per expert). Dropped tokens skip the experts and
    pass through the residual connection.
    """
    capacity = expert_capacity(assignments.numel(), num_experts, capacity_factor)
    kept = torch.zeros_like(assignments, dtype=torch.bool)
    for e in range(num_experts):
        idx = (assignments == e).nonzero(as_tuple=True)[0]
        kept[idx[:capacity]] = True
    dropped = int((~kept).sum().item())
    return kept, dropped, capacity


if __name__ == "__main__":
    torch.manual_seed(0)
    x, w = torch.randn(1024, 64), torch.randn(64, 8)
    w[:, 0] += 1.5; w[:, 1] += 0.8
    a, _ = route_top1(x, w)
    print(counts_per_expert(a, 8).tolist(), [apply_capacity(a, 8, cf)[1] for cf in (1.0, 1.25, 1.5, 2.0)])
