# Chapter 5 — GPU Architecture and Hardware-Aware Decisions

Companion notebook for Chapter 5. It makes the chapter's claims measurable on a real GPU. It runs on a Colab T4 or any recent NVIDIA GPU.

## Notebook

- [`chapter05_companion.ipynb`](chapter05_companion.ipynb)

## What you measure

1. Bandwidth vs compute: a copy is bound by memory, a large matmul is bound by arithmetic.
2. Coalescing: contiguous access is far faster than strided access for the same byte count.
3. Arithmetic intensity and the roofline: where the three RAG stages fall, computed from real shapes.
4. Mixed precision (FP16): prefill against decode. FP16 speeds up the compute-bound prefill matmul and does little for memory-bound decode.
5. Roofline plot: the chapter's roofline with the RAG stages, drawn with the datasheet peaks of the detected GPU (T4, A100 80GB or H100; enter the values for another GPU).

The notebook closes with five short exercises that sweep copy size against bandwidth, stride against slowdown, and place your own model's shapes on the roofline. They are numbered as notebook exercises 1 to 5, separately from the chapter's Exercises 5.1 to 5.8, and Section 6 of the notebook maps each one to the chapter exercise it shortens.

## Profiling commands

The Nsight Systems and Nsight Compute commands in the chapter (Code 5.4) run a script called `rag_decode.py`. Build it
from the decode loop with NVTX ranges in the Chapter 2 lab (`decode_with_nvtx` in
[`../chapter02/chapter02_lab.ipynb`](../chapter02/chapter02_lab.ipynb)) and add `torch.cuda.profiler.start()` and
`torch.cuda.profiler.stop()` around the decode steps, so that `--capture-range=cudaProfilerApi` records only that
region. The Nsight Systems report records the process environment: keep tokens out of it and do not share the file.

## Versions

Tested stack (Chapters 1–5 and 7): Python 3.11, PyTorch 2.14.1 (CUDA 13.0), transformers 5.20.2, sentence-transformers 6.2.0. The book's measurements come from an NVIDIA A100 80GB (reference), an H100 and a T4; the case-study data is in [`case_study/reference_results/`](../case_study/reference_results/). Colab ships its own PyTorch build, so the notebooks do not reinstall torch.

## Run it

Open the notebook in Google Colab, set Runtime > Change runtime type > T4 GPU, and run top to bottom. A GPU runtime is required.

## T4 hardware specifications

The values Chapter 5 uses for occupancy, Roofline and Tensor Core reasoning (NVIDIA T4, Turing, compute capability 7.5). Product figures are theoretical limits; sustained performance is lower and must be measured on the actual system.

| Property | T4 value | Why it matters |
|---|---|---|
| Streaming multiprocessors | 40 | Determines how many blocks can execute concurrently across the device |
| CUDA cores | 2,560 total; 64 FP32 cores per SM | Defines the FP32 scalar/vector arithmetic resource and the per-SM pipeline width |
| Tensor Cores | 320 total; 8 per SM | Accelerate supported FP16 and integer matrix operations |
| Warp schedulers | 4 per SM | Each scheduler manages a static subset of resident warps and can issue one instruction per cycle |
| Maximum resident warps | 32 per SM | The denominator for theoretical and achieved occupancy on Turing |
| Maximum threads and blocks | 1,024 threads and 16 blocks per SM | The other two residency limits in the occupancy calculation |
| Register file | 64K 32-bit registers per SM | Register allocation per thread can limit resident blocks and occupancy |
| Registers per thread | Up to 255 | Kernels that need more spill to local memory |
| Unified L1/shared structure | 96 KB total; supported 64/32 or 32/64 KB carveouts | Shared-memory use and cache carveout affect locality and block residency |
| Shared memory | Up to 64 KB per SM; large per-block allocations require opt-in | Allocated per resident block and can cap concurrent blocks |
| L2 cache | 4 MB, shared by all SMs | Last on-chip level before device memory; small relative to model weights |
| Device memory | 16 GB GDDR6 | Holds model weights, activations, workspace, and KV cache |
| Peak device-memory bandwidth | 320+ GB/s | Provides the theoretical device-memory Roofline slope |
| FP32 peak | 8.1 TFLOP/s | Relevant compute roof for FP32 CUDA-core work |
| FP16 mixed-precision peak | 65 TFLOP/s | Relevant only for Tensor-Core-eligible FP16 operations and appropriate shapes |
| PCIe | Generation 3 x16 | Host-device transfer ceiling; actual throughput depends on payload and pinning |
| Boost clock | 1,590 MHz | Basis of the published peaks; sustained clocks are lower under the 70 W power limit |

## Online resources

The print chapter refers to the material below.

- Knowledge check answers: [`knowledge_check_solutions.md`](knowledge_check_solutions.md).
- The chapter's eight hands-on exercises and the references cited in the chapter, numbered as in the text, are in the
  book's online resources:
  - Exercise 5.1: Occupancy is not utilization
  - Exercise 5.2: Correct coalescing
  - Exercise 5.3: Build empirical roofs
  - Exercise 5.4: Dense-linear AI versus M
  - Exercise 5.5: Mixed precision with evidence
  - Exercise 5.6: KV-cache scaling
  - Exercise 5.7: Bank conflicts
  - Exercise 5.8: Amdahl handoff
- The list of the book's 51 Performance Engineering Principles, with the chapter that states each one:
  [`docs/principles.md`](../docs/principles.md).
