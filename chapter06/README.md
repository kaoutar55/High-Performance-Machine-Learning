# Chapter 6 — Practical High-Performance Kernels with Triton

Companion notebook for Chapter 6. It turns the chapter's claims into runnable, measurable Triton kernels. It runs on any Ampere-or-newer NVIDIA GPU with the pinned Triton build.

## Notebook

- [`chapter06_companion.ipynb`](chapter06_companion.ipynb)

## What you build and measure

1. Vector add: your first kernel — the launch grid, the boundary mask, and a correctness check.
2. Fused softmax: one read and one write instead of many, benchmarked against a naive multi-pass baseline **and** against `torch.nn.functional.softmax`.
3. Fused LayerNorm and RMSNorm: the normalization fusion that libraries like Liger-Kernel ship.
4. Tiled matmul: `tl.dot`, an FP32 accumulator, grouped ordering for L2 reuse, and the honest fraction of cuBLAS you reach.
5. Autotuning: searching block sizes, `num_warps`, and `num_stages`, with first-call cost measured separately from steady state.
6. Debugging: `TRITON_INTERPRET=1` running a kernel on the CPU so you can `print` tile values.

Every kernel is validated against a plain PyTorch reference with `torch.allclose` at a dtype-appropriate tolerance. A fast kernel that computes the wrong answer is worse than the slow one it replaced.

The notebook closes with exercises that sweep block size against bandwidth, compare the matmul to cuBLAS, and register a kernel as a custom PyTorch operator.

## Versions

The chapter pins **PyTorch 2.6 + Triton 3.2** for reproducibility, and its published numbers come from an NVIDIA A100 80GB PCIe (see `chapter06_companion_executed.ipynb`). The kernels use stable API and run across nearby versions; if a cell fails on an API name, check your Triton version first.

## Run it

Open the notebook in Google Colab or on any GPU machine and run top to bottom. On a Colab T4 (Turing), first run `pip install torch==2.6.0 triton==3.2.0`: Triton dropped Turing support in 3.3, so the Triton that ships with a current Colab runtime does not compile for a T4. A GPU runtime is required — Triton compiles for the GPU, and there is no CPU fallback for these kernels except the interpreter used in the debugging section.
