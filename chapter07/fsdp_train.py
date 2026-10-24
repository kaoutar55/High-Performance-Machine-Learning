"""FSDP2 peak-memory study for Chapter 7: measured peak against the modeled 16/N bytes-per-parameter floor.

A stack of transformer encoder layers (about 407M parameters) is sharded with fully_shard and trained in BF16
with FP32 gradient reduction (MixedPrecisionPolicy). --checkpoint wraps every layer in activation checkpointing.

    torchrun --standalone --nproc_per_node=2 fsdp_train.py --steps 50 --warmup 15
    torchrun --standalone --nproc_per_node=2 fsdp_train.py --steps 50 --warmup 15 --checkpoint

Needs PyTorch 2.6 or later (torch.distributed.fsdp.fully_shard).
"""
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import checkpoint


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, use_ckpt: bool):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                                batch_first=True, norm_first=True)
        self.use_ckpt = use_ckpt

    def forward(self, x):
        if self.use_ckpt and self.training:
            return checkpoint(self.layer, x, use_reentrant=False)
        return self.layer(x)


class Model(nn.Module):
    def __init__(self, d_model=2048, n_heads=16, n_layers=8, out_dim=2000, use_ckpt=False):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d_model, n_heads, use_ckpt) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=4, help="sequences per GPU per step")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--checkpoint", action="store_true")
    args = ap.parse_args()

    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)

    torch.manual_seed(0)
    model = Model(use_ckpt=args.checkpoint)
    n_params = sum(p.numel() for p in model.parameters())
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for b in model.blocks:
        fully_shard(b, mp_policy=mp)
    fully_shard(model, mp_policy=mp)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)  # FP32 master weights and moments, sharded

    modeled_gb = n_params * 16 / world / 1e9
    if rank == 0:
        print(f"[rank 0] params={n_params / 1e6:.1f}M  world={world}  checkpoint={args.checkpoint}", flush=True)
        print(f"[rank 0] modeled model-state/GPU (16B/param / world) = {modeled_gb:.3f} GB", flush=True)

    g = torch.Generator(device=dev).manual_seed(rank)
    x = torch.randn(args.batch_size, args.seq_len, 2048, device=dev, generator=g)
    y = torch.randn(args.batch_size, args.seq_len, 2000, device=dev, generator=g)
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = None
    for step in range(args.steps):
        if step == args.warmup:
            torch.cuda.synchronize(dev); dist.barrier(); t0 = time.perf_counter()
        loss = nn.functional.mse_loss(model(x).float(), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - t0

    peak = torch.tensor([torch.cuda.max_memory_allocated(dev) / 1e9], device=dev)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    agg = (args.steps - args.warmup) * args.batch_size * world / elapsed
    if rank == 0:
        p = peak.item()
        print(f"[rank 0] measured peak allocated/GPU = {p:.3f} GB")
        print(f"[rank 0] gap (peak - modeled state) = {p - modeled_gb:.3f} GB "
              f"(activations + transient all-gather + fragmentation)")
        print(f"[rank 0] aggregate samples/s = {agg:.1f}")
        print(f"RESULT world_size={world} params_m={n_params / 1e6:.1f} checkpoint={args.checkpoint} "
              f"modeled_state_gb={modeled_gb:.4f} peak_gb={p:.4f} aggregate_samples_s={agg:.2f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
