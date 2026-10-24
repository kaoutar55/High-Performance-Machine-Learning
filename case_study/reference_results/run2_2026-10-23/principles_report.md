# Principles across GPUs

Each row is a claim the book teaches. The evidence comes from this run's measurements.

| Principle | Chapters | A100-80GB | H100 | T4 |
|---|---|---|---|---|
| At batch 1 every decode GEMM is a matrix-vector product (~1 FLOP/byte), so memory bandwidth sets the Roofline ceiling | Ch 1, 2, 5 | holds | holds | holds |
| Eager decode pays a roughly fixed CPU and launch cost per step; removing it helps in proportion to that cost's share of the step | Ch 2, 3, 4 | holds | holds | holds |
| A decode batch reads the weights once per step, so tokens/s rises almost linearly with batch size until compute or KV-cache traffic takes over | Ch 5 | holds | holds | holds |
| Prefill is compute-bound; decode is memory-bound | Ch 1, 5 | holds | holds | holds |
| Utilization grows with the rows per matrix multiply until a hardware-dependent knee | Ch 1, 3, 5 | holds | holds | holds |
| In this case study's configuration, decode dominates the RAG pipeline's latency | Ch 1, 2 | holds | holds | holds |
| 16-bit Tensor Core math gives several times FP32 throughput on compute-bound work | Ch 3, 5 | holds | holds | holds |
| torch.compile(model) compiles forward only; generate()'s loop stays eager | Ch 4 | holds | holds | holds |
| SDPA dispatches at run time to the first eligible fused attention backend | Ch 2, 3, 5 | holds | holds | holds |

## Evidence

### At batch 1 every decode GEMM is a matrix-vector product (~1 FLOP/byte), so memory bandwidth sets the Roofline ceiling
_A decode step at batch 1 reads every weight once to produce one token, about 1 FLOP per byte,
so its ceiling is memory bandwidth: it can use only a tiny fraction of peak compute._

Sources: Williams, Waterman, Patterson, Roofline, CACM 2009; NVIDIA, GPU Performance Background User's Guide

- **A100-80GB** (holds): 17.5 ms/token uses 0.09% of peak compute; weight-read floor 2.5 ms/token, reached at 14%
- **H100** (holds): 11.8 ms/token uses 0.04% of peak compute; weight-read floor 1.5 ms/token, reached at 13%
- **T4** (holds): 24.6 ms/token uses 0.31% of peak compute; weight-read floor 15.7 ms/token, reached at 64%

### Eager decode pays a roughly fixed CPU and launch cost per step; removing it helps in proportion to that cost's share of the step
_Whether decode reaches its bandwidth ceiling depends on per-step overhead (Python, kernel
launches). Removing that overhead (static cache + compilation / CUDA graphs) helps in proportion
to the overhead's share of the step: a lot on fast GPUs, little on a GPU already near its floor._

Sources: PyTorch blog, Accelerating Generative AI with PyTorch II: GPT, Fast (2023); Ansel et al., PyTorch 2, ASPLOS 2024

- **A100-80GB** (holds): eager 17.5 ms/token, floor 2.5 ms, so overhead is 86% of the step; hf_autocompile_static_cache gives 6.0 ms/token (2.91x)
- **H100** (holds): eager 11.7 ms/token, floor 1.5 ms, so overhead is 87% of the step; hf_autocompile_static_cache gives 3.5 ms/token (3.33x)
- **T4** (holds): eager 24.7 ms/token, floor 15.7 ms, so overhead is 37% of the step; hf_autocompile_static_cache gives 23.5 ms/token (1.05x)

### A decode batch reads the weights once per step, so tokens/s rises almost linearly with batch size until compute or KV-cache traffic takes over
_Batching amortizes weight reads: decode throughput grows almost in proportion to batch size
until compute or KV-cache traffic takes over._

Sources: Agrawal et al., Sarathi-Serve, OSDI 2024; Pope et al., Efficiently Scaling Transformer Inference, MLSys 2023; Kwon et al., PagedAttention, SOSP 2023

- **A100-80GB** (holds): batch 16 gives 14.9x the tokens/s of batch 1 (ideal 16x)
- **H100** (holds): batch 16 gives 14.9x the tokens/s of batch 1 (ideal 16x)
- **T4** (holds): batch 16 gives 11.2x the tokens/s of batch 1 (ideal 16x)

### Prefill is compute-bound; decode is memory-bound
_Prefill processes every prompt token per weight read, so it runs far closer to peak compute
than decode does._

Sources: Agrawal et al., Sarathi-Serve, OSDI 2024; Patel et al., Splitwise, ISCA 2024

- **A100-80GB** (holds): prefill 119.4 TFLOP/s vs decode 0.29 TFLOP/s (416x)
- **H100** (holds): prefill 187.4 TFLOP/s vs decode 0.42 TFLOP/s (442x)
- **T4** (holds): prefill 24.3 TFLOP/s vs decode 0.20 TFLOP/s (119x)

### Utilization grows with the rows per matrix multiply until a hardware-dependent knee
_GPU utilization depends on how many rows each matrix multiply processes. Decode has one row per
sequence, so batching sequences helps almost linearly. An embedder chunk already supplies hundreds
of token rows, so batching chunks helps less; how much less depends on where the GPU's knee is._

Sources: NVIDIA, Linear/Fully-Connected Layers User's Guide; Agrawal et al., Sarathi-Serve, OSDI 2024

- **A100-80GB** (holds): decode (1 row/sequence): batch 16 = 14.9x batch 1; embedder (477 token rows/chunk): batch 32 = 4.1x batch 1
- **H100** (holds): decode (1 row/sequence): batch 16 = 14.9x batch 1; embedder (477 token rows/chunk): batch 32 = 4.9x batch 1
- **T4** (holds): decode (1 row/sequence): batch 16 = 11.2x batch 1; embedder (477 token rows/chunk): batch 32 = 1.2x batch 1

### In this case study's configuration, decode dominates the RAG pipeline's latency
_In this case study (small in-memory index, one retrieval, 2B generator, 20 output tokens),
token-by-token decode dominates end-to-end latency. At billion-scale retrieval or with long
contexts encoded online, retrieval or encoding can dominate instead (RAGO, Shen et al.)._

Sources: Shen et al., arXiv:2412.11854 (retrieval can be 41% of latency); Jiang et al., RAGO, arXiv:2503.14649 (the dominant stage depends on configuration)

- **A100-80GB** (holds): decode 339 ms of 382 ms stage total (89%)
- **H100** (holds): decode 196 ms of 223 ms stage total (88%)
- **T4** (holds): decode 458 ms of 586 ms stage total (78%)

### 16-bit Tensor Core math gives several times FP32 throughput on compute-bound work
_Running the encoder in 16-bit on Tensor Cores raises throughput substantially over FP32._

Sources: Micikevicius et al., Mixed Precision Training, ICLR 2018; PyTorch CUDA semantics: TF32 notes

- **A100-80GB** (holds): FP16 274 vs FP32 52 chunks/s at batch 32 (5.2x)
- **H100** (holds): FP16 511 vs FP32 119 chunks/s at batch 32 (4.3x)
- **T4** (holds): FP16 57 vs FP32 12 chunks/s at batch 32 (4.8x)

### torch.compile(model) compiles forward only; generate()'s loop stays eager
_torch.compile speeds up only code that runs through the compiled function: compiling the module
and then calling .generate() leaves the decode loop, which generate() drives, essentially unchanged._

Sources: Hugging Face, LLM optimization overview (compile inside generate); PyTorch blog, GPT, Fast (compile the per-token decode step)

- **A100-80GB** (holds): eager 17.5 ms/token; compile-module-then-generate 17.8
- **H100** (holds): eager 11.7 ms/token; compile-module-then-generate 11.1
- **T4** (holds): eager 24.7 ms/token; compile-module-then-generate 24.7

### SDPA dispatches at run time to the first eligible fused attention backend
_Attention runs as one fused kernel on every GPU; which kernel PyTorch's SDPA dispatcher picks
(FlashAttention, memory-efficient, cuDNN) depends on GPU, dtype, mask and library versions, so read
it from the profiler._

Sources: Dao et al., FlashAttention, NeurIPS 2022; PyTorch 2.5 release blog (cuDNN SDPA on H100)

- **A100-80GB** (holds): compute capability 8.0: mem_efficient (fmha_cutlassF_f16_aligned_64x64_rf_sm80(PyTorchMemEffAttenti)
- **H100** (holds): compute capability 9.0: cudnn (cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_)
- **T4** (holds): compute capability 7.5: mem_efficient (fmha_cutlassF_f16_aligned_64x64_rf_sm75(PyTorchMemEffAttenti)

## Platforms

- **A100-80GB**: NVIDIA A100-SXM4-80GB (CC 8.0), PyTorch 2.14.1+cu130, CUDA 13.0, transformers 5.20.2
- **H100**: NVIDIA H100 80GB HBM3 (CC 9.0), PyTorch 2.14.1+cu130, CUDA 13.0, transformers 5.20.2
- **T4**: Tesla T4 (CC 7.5), PyTorch 2.14.1+cu130, CUDA 13.0, transformers 5.20.2
