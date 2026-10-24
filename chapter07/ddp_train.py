"""DistributedDataParallel scaling study for Chapter 7.

Trains a small MLP stack on synthetic data and reports aggregate throughput (samples/s), peak memory, and a
check that every rank holds the same gradients after backward (the all-reduce worked).

    torchrun --standalone --nproc_per_node=1 ddp_train.py --steps 60 --warmup 15 --batch-size 32 --d-model 1024
    torchrun --standalone --nproc_per_node=2 ddp_train.py --steps 60 --warmup 15 --batch-size 32 --d-model 1024

--no-sampler drops the DistributedSampler, so every rank trains on the same samples: the run still
"scales", but the ranks duplicate work. It is there to show the bug, not as a configuration to use.
"""
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


def build_model(d_model: int, blocks: int = 2) -> nn.Module:
    layers = []
    for _ in range(blocks):
        layers += [nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)]
    return nn.Sequential(*layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32, help="samples per GPU per step")
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--no-sampler", action="store_true")
    args = ap.parse_args()

    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)

    # Synthetic regression data: enough samples for `steps` batches on every rank.
    g = torch.Generator().manual_seed(0)
    n = args.steps * args.batch_size * world
    x = torch.randn(n, args.d_model, generator=g)
    y = torch.randn(n, args.d_model, generator=g)
    ds = TensorDataset(torch.arange(n), x, y)
    if args.no_sampler:
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=True)
    else:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=0)
        sampler.set_epoch(0)
        idx = list(iter(sampler))
        print(f"[rank {rank}] shard size={len(idx)} first={idx[0]} last={idx[-1]}", flush=True)
        loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, drop_last=True)

    torch.manual_seed(0)  # identical initial weights on every rank (DDP also broadcasts rank 0's)
    model = DDP(build_model(args.d_model).to(dev), device_ids=[local_rank])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    torch.cuda.reset_peak_memory_stats(dev)
    t0 = None
    timed = 0
    checked = False
    for step, (_, xb, yb) in enumerate(loader):
        if step == args.warmup:
            torch.cuda.synchronize(dev); dist.barrier(); t0 = time.perf_counter()
        xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
        loss = loss_fn(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()                       # DDP all-reduces gradient buckets during backward
        if not checked:
            flat = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
            fingerprint = torch.stack([flat.sum(), flat.abs().sum(), flat.pow(2).sum()]).double()
            gathered = [torch.zeros_like(fingerprint) for _ in range(world)]
            dist.all_gather(gathered, fingerprint)
            same = all(torch.allclose(gathered[0], t, rtol=1e-6, atol=0) for t in gathered)
            if rank == 0:
                print(f"[rank 0] gradients identical across ranks after backward: {same}", flush=True)
            checked = True
        opt.step()
        if t0 is not None:
            timed += 1
        if step + 1 >= args.steps:
            break
    torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - t0

    rate = torch.tensor([timed * args.batch_size / elapsed], device=dev)
    rates = [torch.zeros_like(rate) for _ in range(world)]
    dist.all_gather(rates, rate)
    peak_gb = torch.cuda.max_memory_allocated(dev) / 1e9
    if rank == 0:
        per_rank = [f"{r.item():.1f}" for r in rates]
        agg = sum(r.item() for r in rates)
        print(f"[rank 0] per-rank samples/s: {per_rank}")
        print(f"[rank 0] aggregate samples/s (world={world}): {agg:.1f}")
        print(f"[rank 0] peak allocated: {peak_gb:.3f} GB")
        print(f"RESULT world_size={world} batch_per_gpu={args.batch_size} aggregate_samples_s={agg:.2f} "
              f"peak_gb={peak_gb:.4f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
