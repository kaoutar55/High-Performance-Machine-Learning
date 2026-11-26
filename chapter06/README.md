# Chapter 6: Practical High-Performance Kernels with Triton

Companion notebook for Chapter 6. It contains the chapter's Triton kernels, checks each one against a PyTorch
reference and times them with CUDA events.

## Notebook

- [`chapter06_companion_2p14.ipynb`](chapter06_companion_2p14.ipynb): the companion notebook named in the chapter,
  pinned to PyTorch 2.14.1 (CUDA 13.0 build) and Triton 3.8.0.

The earlier notebook for PyTorch 2.6.0 and Triton 3.2.0, `chapter06_companion.ipynb`, and its executed copy
`chapter06_companion_executed.ipynb` do not match the current chapter text.

## What you build and measure

1. Vector add: the launch grid, the boundary mask and a correctness check.
2. Fused softmax: one read and one write per row, timed against a naive multi-pass baseline and against
   `torch.nn.functional.softmax`.
3. Fused LayerNorm and RMSNorm, a fused QuickGELU-plus-dropout kernel with an on-chip random number generator, and
   the RMSNorm kernel registered as a custom operator with a Triton backward.
4. Tiled matmul: `tl.dot`, an FP32 accumulator and grouped ordering for L2 reuse, timed against cuBLAS.
5. Autotuning: a search over block sizes, `num_warps` and `num_stages`, with the first-call cost measured
   separately from the steady state.
6. Debugging: `TRITON_INTERPRET=1` running a kernel on the CPU so you can `print` tile values.

Each kernel is checked against a plain PyTorch reference at a dtype-appropriate tolerance, most of them with
`torch.testing.assert_close` and the fused softmax with `torch.allclose`. The notebook closes with Exercises 6.1 to
6.9 from the chapter.

## Versions

The notebook is pinned to **PyTorch 2.14.1 and Triton 3.8.0**. The chapter's reference measurements were taken on
an NVIDIA H100 80GB HBM3 on 24 November 2026; `results/ch6_2p14_h100_2026-11-24.json` holds them. The setup cell
stops with an error when it finds a different PyTorch and Triton pair, unless you set `CH6_ALLOW_OTHER_STACK=1`.
Timings differ between GPUs and software releases.

## Run it

Install the pinned pair, then run the notebook top to bottom on a GPU machine:

```bash
pip install torch==2.14.1 triton==3.8.0 --index-url https://download.pytorch.org/whl/cu130
```

Triton lists NVIDIA GPUs of compute capability 8.0 (Ampere) and newer as supported; the Turing-generation T4
(compute capability 7.5) is below that line, and the notebook is tested on compute capability 8.0 and newer. A GPU
is required: there is no CPU path for these kernels except the interpreter used in the debugging section.

## Online resources

The print chapter refers to the material below. Release details change; check each item against the release you
run.

### Kernel languages around Triton: release details

The section "Where Triton fits among kernel languages" names Helion, Gluon, NVIDIA's cuTile Python and the CuTe DSL
(CuTeDSL). Release details:

- cuTile Python is part of NVIDIA CUDA Tile, which NVIDIA introduced with CUDA 13.1 (NVIDIA Technical Blog,
  ["NVIDIA CUDA 13.1 Powers Next-Gen GPU Programming with NVIDIA CUDA Tile and Performance Gains"](https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains/),
  December 2025). The first release supports compute capabilities 8.x, 10.x, 11.x and 12.x.
- The CuTe DSL is the first of the CUTLASS Python DSLs and was released with CUTLASS 4.0; NVIDIA describes it as a
  public beta ([CUTLASS documentation, "Overview"](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/overview.html)).

### Enabling the CuTeDSL GEMM backend in TorchInductor

In the PyTorch releases described in the PyTorch blog post "Generating State-of-the-Art GEMMs with TorchInductor's
CuTeDSL Backend" (April 7, 2026), the CuTeDSL GEMM path is opt-in. It is enabled in max-autotune mode with

```python
torch._inductor.config.max_autotune_gemm_backends = "ATEN,TRITON,NVGEMM"
```

and NVIDIA's CuTeDSL packages installed. Check the option name and the package requirements against the PyTorch
release you use.
