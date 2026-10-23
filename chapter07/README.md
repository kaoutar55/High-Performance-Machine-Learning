# Chapter 7 — Scaling Out: Choosing a Distributed Training Strategy

Companion notebook for Chapter 7. It reproduces the chapter's quantitative results in two kinds of cell: analytical cells that compute the chapter's models live on any machine, and measured cells that report DDP and FSDP runs on two GPUs.

## Notebook

- [`chapter07_companion.ipynb`](chapter07_companion.ipynb)
- [`results/modal_results_2xa10g.txt`](results/modal_results_2xa10g.txt): raw output of the measured runs

## What you compute and measure

1. The memory wall: model-state bytes per parameter for each training regime, and when a model stops fitting on one GPU.
2. Communication cost: ring all-reduce against a parameter server, using the α–β cost model.
3. Pipeline bubble fraction against the number of microbatches.
4. Tensor parallelism: a numerical check that Megatron's column-then-row split of the MLP gives the same result as the unsplit layer.
5. Mixture-of-experts routing: load imbalance and capacity dropping under a biased gate.
6. Measured DDP scaling on two GPUs, for a communication-bound and a compute-heavier model.
7. Measured FSDP peak memory against the modeled 16/N bytes-per-parameter floor, with and without activation checkpointing.

## Hardware

The measured cells were run on **two NVIDIA A10G GPUs (24 GB each, PCIe, no NVLink)** on Modal. The analytical cells run on CPU. On an NVLink node the DDP scaling numbers are higher; the chapter explains why the interconnect sets the ceiling.

## Versions

Tested stack for Chapters 1–5 and 7: Python 3.11, PyTorch 2.14.1 (CUDA 13.0). FSDP2 (`torch.distributed.fsdp.fully_shard`) needs PyTorch 2.6 or later.

## Run it

The analytical cells run anywhere, including a CPU laptop or a Colab CPU runtime. They import the chapter's helper modules (`mem_wall`, `comm_cost`, `bubble`, `tp_mlp`, `moe_routing`), which must sit next to the notebook.

The measured cells need at least two GPUs. The notebook gives the exact `torchrun` commands for `ddp_train.py` and `fsdp_train.py`, and a Modal entry point (`modal_run.py`) that runs both studies on a two-GPU container.

> **Status:** the helper modules and the three launch scripts (`ddp_train.py`, `fsdp_train.py`, `modal_run.py`) are not yet in this folder. Until they are added, the notebook's first code cell fails on import, and the measured numbers are reproducible only from the recorded output in `results/`.
