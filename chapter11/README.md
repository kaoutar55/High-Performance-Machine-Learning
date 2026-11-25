# Chapter 11: Navigating the Accelerator and Compiler Landscape

**Status: case study run on 2026-11-15 to 2026-11-23.** The measured results are in `results/`.

Companion code for Chapter 11. The case study runs the RAG pipeline's two generators, Gemma-2B-IT and
Llama-3.1-8B-Instruct, on NVIDIA A100 and H100 GPUs and on Google Cloud TPU v5e and v6e. For each configuration it
measures prefill time, time per output token at batch size 1 and numerical agreement with an FP32 CPU reference,
with PyTorch eager, `torch.compile`, vLLM 0.33.0 (BF16 and FP8 weights), TorchTPU and JAX. Separate experiments
measure batch invariance (determinism) in vLLM, ONNX Runtime against PyTorch, and how DTensor and JAX place the
collectives of a sharded matrix multiplication.

## Scripts

| Script | What it does |
|---|---|
| `common.py` | Shared settings: prompts, prompt length, number of new tokens, environment record |
| `jax_basics.py` | Code 11.1: a two-layer MLP in JAX with `grad`, `vmap` and `jit` (CPU, GPU or TPU) |
| `sharding_compare.py` | Code 11.2: the same row-sharded weight in PyTorch DTensor (`torchrun`, 2 GPUs) and in JAX |
| `colab/gpu_reference_v3.py` | GPU rows (eager, `torch.compile`, vLLM BF16 and FP8), median of repeats, top-1 agreement |
| `colab/determinism_colab.py` | Batch-invariance experiment: default and `VLLM_BATCH_INVARIANT=1` at 1, 16 and 64 concurrent requests |
| `colab/*.ipynb` | Colab notebooks for the GPU rows, the TPU rows and a GPU check |
| `h100/modal_h100_all.py` | All H100 rows and the determinism experiment on Modal in one command (`gpu="H100!"` pins the H100) |
| `h100/modal_ch11_fixups.py` | ONNX Runtime with and without IO binding, and the DTensor check on 2 H100s |
| `h100/modal_jax_shard_check.py` | The JAX half of Code 11.2 on 8 simulated CPU devices |
| `h100/modal_reference.py` | Earlier single-GPU reference script (kept for the commands in its docstring) |
| `tpu/torchtpu_gemma.py` | Code 11.3: TPU rows with TorchTPU (`torch.compile(backend="tpu")`) |
| `tpu/jax_gemma.py`, `tpu/run_jax.sh` | JAX rows on a TPU with KerasHub |
| `onnx/export_and_run.py` | Code 11.4: `torch.export` to ONNX and ONNX Runtime with the CUDA execution provider |
| `parity/cpu_reference.py` | FP32 CPU reference: greedy tokens and teacher-forced logits |
| `parity/compare_logits.py` | Code 11.5: KL divergence and top-1 agreement of a platform's logits with the reference |
| `determinism/run_determinism.py` | Stand-alone version of the batch-invariance experiment |
| `amd/bench_vllm.py`, `amd/mi300x_vllm.sh` | Optional MI300X rows with vLLM on ROCm (not part of the measured case study) |

## Results

| File | Platform and content |
|---|---|
| `results/colab_a100_r3_2026-11-15.json` | A100 80GB (Colab): both models, eager, `torch.compile`, vLLM; median of 3 |
| `results/colab_a100_determinism_2026-11-15.json` | A100: batch invariance, default and invariant mode |
| `results/modal_h100_2026-11-15*.json` | H100 80GB SXM (Modal): both models, all stacks, FP8, determinism |
| `results/modal_fixups_2026-11-16_0109.json` | H100: ONNX Runtime timings with outputs on the host and on the GPU; DTensor collectives |
| `results/cluster_h100_2026-11-23/` | H100 80GB SXM, GPU cluster node (2 x Intel Xeon Platinum 8468, 16 cores): both models, eager, `torch.compile`, vLLM BF16 and FP8, median of 3, with logs. Source of the H100 rows in Tables 11.8 to 11.10; the two Modal sessions show how much the eager and compiled rows depend on the host |
| `results/colab_tpu_v6e1_*.json`, `results/colab_tpu_v5e1_*.json` | TPU v6e and v5e (Colab): TorchTPU rows and repeats |
| `results/colab_tpu_v5e1_jax_2026-11-16.json` | TPU v5e: JAX rows and the precision comparison |
| `results/cluster_jax_shard_2026-11-19.json` | JAX sharding check on 8 CPU devices: output sharding and the collectives in the compiled program |

Each result file records the GPU or TPU, the driver and the package versions.

## What to expect when you reproduce the results

The GPU and TPU rows are medians of three repeats within one session, and decode times stayed within 4% of the median
on the GPUs. Between sessions and setups, the eager and `torch.compile` rows varied much more, because at batch size 1
they are limited by CPU-side overhead. With the same script, Gemma-2B eager on the H100 took 11.3 and 18.7 ms per
output token in two Modal sessions on different hosts, and 11.5 ms on a GPU cluster node with a known CPU; Chapter 8
reports 16.5 ms on the A100 for the FP16 model with 20 output tokens, against 25.2 ms in Table 11.8. The vLLM rows
agreed with Chapter 8 within 2% for Gemma-2B and 5% for Llama-3.1-8B. Expect the eager and compiled rows to differ
from the book by up to a factor of two and the vLLM rows by a few percent; compare the ordering of the stacks
and the ratio of each value to the bandwidth bound.

## Stack

vLLM 0.33.0, PyTorch 2.14, transformers 5.x, ONNX Runtime 1.32 (CUDA execution provider); JAX and TorchTPU on Colab
TPU runtimes. Gated models (Gemma, Llama) need a Hugging Face token: on Modal a secret named `huggingface`, on Colab
a Colab secret.

## Online resources

The print chapter refers to the material below. Reference numbers are those of the chapter.

### Selected accelerators (November 2026)

Vendor-stated dense peak rates per chip, memory and scale-up links. A dash means the format is not supported or not stated. TPU ICI figures are bidirectional. Cerebras quotes its peak with sparsity; the WSE-3 value is an estimated dense rate. Sources: references [15], [16] and [19] to [25] of the chapter.

| Accelerator | BF16 TFLOP/s | FP8 TFLOP/s | FP4 TFLOP/s | Memory | Bandwidth | Scale-up link |
|---|---|---|---|---|---|---|
| NVIDIA A100 SXM | 312 | – | – | 80 GB HBM2e | 2.04 TB/s | NVLink 600 GB/s |
| NVIDIA H100 SXM | 989 | 1,979 | – | 80 GB HBM3 | 3.35 TB/s | NVLink 900 GB/s |
| NVIDIA B200 (HGX) | 2,250 | 4,500 | 9,000 | about 180 GB HBM3e | 8 TB/s | NVLink 1.8 TB/s |
| AMD MI355X | 2,517 | 5,033 | 10,066 (MXFP4) | 288 GB HBM3E | 8 TB/s | Infinity Fabric |
| Google TPU v5e | 197 | – (INT8 393) | – | 16 GB HBM | 0.82 TB/s | ICI 400 GB/s |
| Google TPU v6e | 918 | – (INT8 1,836) | – | 32 GB HBM | 1.64 TB/s | ICI 800 GB/s |
| Google TPU7x (Ironwood) | 2,307 | 4,614 | – | 192 GiB HBM | 7.4 TB/s | ICI 1.2 TB/s |
| AWS Trainium3 | 671 | 2,517 (MXFP8) | 2,517 (MXFP4) | 144 GiB HBM | 4.9 TB/s | NeuronLink-v4 |
| Cerebras WSE-3 | about 12,500 (FP16, estimated dense) | – | – | 44 GB on-wafer SRAM | about 21 PB/s | System fabric |
| Groq 3 LP30 | – | 1,200 | – | about 500 MB SRAM | about 150 TB/s | 256 chips per LPX rack |

### Execution models of current accelerators

| Property | GPU (SIMT) | Systolic, compiler-scheduled | Spatial dataflow |
|---|---|---|---|
| Examples | NVIDIA, AMD Instinct | TPU, Trainium, Gaudi | Groq LPU (language processing unit), Cerebras WSE, SambaNova RDU |
| Who schedules | Hardware at run time | Compiler, with hardware queues | Compiler, fully static |
| Where weights live during decode | HBM | HBM | On-chip SRAM (SambaNova: SRAM, HBM and DDR) |
| Shapes | Dynamic | Static or bucketed | Static |
| Run-to-run timing | Variable | Mostly predictable | Predictable |
| Strength | Generality, software ecosystem | Efficiency on large matrix work | Latency at small batch |
| Main limit | Overhead per instruction | Compile time, shape changes | Capacity per chip, compile time |

### Disaggregation by layer type (NVIDIA LPX)

NVIDIA's LPX configuration, announced in March 2026, is the first commercial example: GPUs run prefill and decode
attention, and Groq-derived LP30 chips run the feed-forward layers, with NVIDIA Dynamo orchestrating the two [25].

![Disaggregation by layer type in NVIDIA's LPX configuration](online/lpx_disaggregation.png)

Disaggregation by layer type in NVIDIA's LPX configuration, as described by NVIDIA. Prefill and decode attention run on GPUs, where the KV cache is held in HBM; the feed-forward layers of decode run on SRAM-based LPUs; activations move between them at each layer. Source: reference [25].

### Sources of nondeterminism and their controls

| Source | Varies between | Control | Cost |
|---|---|---|---|
| Atomic accumulation | Identical runs | torch.use_deterministic_algorithms(True), CUBLAS_WORKSPACE_CONFIG | Slower kernels, errors where no deterministic kernel exists |
| Batch-dependent kernels | Runs with different load | Batch-invariant kernels (vLLM VLLM_BATCH_INVARIANT=1) | Lower throughput; some fast paths disabled |
| Collective algorithm | Message sizes, library versions | Fixed algorithm and versions; batch-invariant mode | Possibly slower collectives |
| Precision defaults | Platforms and frameworks | Explicit precision (TF32 off, precision=HIGHEST in JAX) | Up to several times slower FP32 matrix multiplications |
| FP8 variant | Vendors | Convert checkpoints and rescale | One conversion step |

### Exchange formats, compilers and runtimes (November 2026)

| Tool | Input | Targets | Typical use |
|---|---|---|---|
| ONNX Runtime | ONNX graphs | CPUs, NVIDIA, AMD, Intel, Apple, Qualcomm via execution providers | Portable inference of fixed graphs |
| XLA (OpenXLA) | StableHLO | TPUs, GPUs, CPUs, PJRT plugins | JAX and TorchTPU compilation |
| IREE | StableHLO, other MLIR dialects | CPUs, Vulkan, CUDA, ROCm, accelerators | Embedded and mobile deployment |
| TensorRT-LLM | Hugging Face checkpoints | NVIDIA GPUs | High-throughput NVIDIA serving |
| AOTInductor | torch.export programs | CPUs, GPUs | Python-free PyTorch deployment |
| ExecuTorch | torch.export programs | Phones, microcontrollers, NPUs | On-device inference |
| Apache TVM | Several frameworks | CPUs, GPUs, accelerators | Research, custom hardware |

### Debugging and profiling tools and switches by platform

| Platform | Profiler | Compiler diagnostics | Useful switches |
|---|---|---|---|
| NVIDIA GPU | Nsight Systems, Nsight Compute, PyTorch profiler | TORCH_COMPILE_DEBUG=1, TORCH_LOGS | CUDA_LAUNCH_BLOCKING=1, NVIDIA_TF32_OVERRIDE=0 |
| AMD GPU (ROCm) | rocprofv3 (ROCprofiler-SDK), PyTorch profiler | TORCH_COMPILE_DEBUG=1 | HIP_LAUNCH_BLOCKING=1 |
| Google TPU (JAX, TorchTPU) | XProf in TensorBoard | XLA_FLAGS=--xla_dump_to=DIR | TORCH_TPU_LAUNCH_BLOCKING=1, jax_default_matmul_precision |
| AWS Trainium | Neuron profiler (Neuron Explorer) | Neuron compiler logs | Neuron runtime log level |
| ONNX Runtime | Session profiling (JSON trace) | Graph optimization level settings | log_severity_level=0 for provider assignment |
| vLLM (any backend) | Engine metrics, PyTorch profiler | Attention backend selection logs | VLLM_LOGGING_LEVEL=DEBUG |

### Number formats by hardware generation

The print chapter summarizes this material in one paragraph and refers to Chapter 9. References [4] to [15].

-   FP16 and mixed-precision training (2017). The V100's Tensor Cores multiply FP16 values and accumulate in FP32. FP16's 5-bit exponent gives a normal range of 6.1 × 10⁻⁵ to 65,504, too narrow for many activation gradients, so mixed-precision training keeps FP32 master weights and scales the loss to move the gradients into range [4], the pattern Chapter 5 shows in PyTorch.
-   BF16. Google Brain's bfloat16 keeps FP32's 8-bit exponent and cuts the fraction to 7 bits. It has FP32's range, so training needs no loss scaling, and models across vision, speech, language and recommendation reached FP32 accuracy with unchanged hyperparameters [5]. NVIDIA added BF16 with the A100 in 2020; the T4 has none, so a BF16-trained model runs there in FP16, where activations above 65,504 overflow.
-   TF32 (2020). The A100 also introduced TF32, a Tensor Core mode with no storage format: FP32 inputs are rounded to a 10-bit fraction, multiplied exactly and accumulated in FP32 [6]. It gives FP32 code Tensor Core speed, 156 against 19.5 TFLOP/s on the A100, without saving memory. PyTorch enables it with torch.backends.cuda.matmul.fp32_precision = "tf32", while JAX by default uses TF32 on GPUs and BF16 on TPUs for float32 matrix multiplications [7], a difference that the Portability and Performance section and the case study return to.
-   INT8 and INT4 for inference. Google's first TPU, deployed in 2015, ran inference with 65,536 8-bit integer multiply-accumulate units [8]. NVIDIA's Turing T4 (2018) added INT8 and INT4 Tensor Cores at two and four times its FP16 rate [9]. Integer formats suit inference because the ranges of a trained model's tensors can be measured in advance and stored as scales.
-   FP8 (2022). Hopper GPUs added FP8 in two variants: E4M3 for weights and activations, which need precision, and E5M2 for gradients, which need range [10]. Unlike INT8, FP8 gives each value its own exponent within the tensor's scale, which suits the long-tailed activations of LLMs, and it is now used for training at scale: DeepSeek-V3 was trained in FP8 with fine-grained block scales [11].
-   Microscaling and FP4 (2023–2025). Below 8 bits, one scale per tensor cannot cover the range of the data, because FP4 spans only from 1 to 6 in its normal values. The OCP Microscaling (MX) specification, from AMD, Arm, Intel, Meta, Microsoft, NVIDIA and Qualcomm, attaches a shared power-of-two scale to every block of 32 elements in FP8, FP6, FP4 or INT8 [12][13]. NVIDIA's NVFP4 uses blocks of 16 values with an FP8 scale [14]. Blackwell GPUs execute the MX formats and NVFP4, AMD's MI355X executes MXFP6 and MXFP4, and AWS Trainium3 executes MXFP8 and MXFP4 [15].

### Apple M5 and MLX: prefill and decode follow different roofs

The same reasoning applies to laptops and phones, whose stacks are ExecuTorch, LiteRT, Core ML and Apple's MLX. Apple's measurements of its M5 chip, which added matrix units to the GPU, show the split between compute-bound and memory-bound phases. For Qwen3 and gpt-oss-20b models in MLX, the time to first token improved by 3.3 to 4.1 times over the M4, because prefill is compute-bound and the new matrix units raised the compute, while token generation improved by 19 to 27%, close to the 28% increase in memory bandwidth from 120 to 153 GB/s [33]. The improvement in each phase followed the roof that bounds it, as the Roofline model predicts.

### Weight-read bound for an announced product (NVIDIA Rubin)

Vendor-stated bandwidth. NVIDIA announced the Rubin GPU with up to 22 TB/s of HBM4 bandwidth (NVIDIA Technical Blog, "Inside the NVIDIA Rubin Platform: Six New Chips, One AI Supercomputer", 5 January 2026, https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/). The Vera Rubin NVL72 product page lists 288 GB of HBM4 at 19.2 TB/s per GPU (NVIDIA, "NVIDIA Vera Rubin NVL72", accessed 25 November 2026, https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/). The bound is the bandwidth divided by the 16 GB read per token for Llama-3.1-8B in BF16. The printed Table 11.2 and Figures 11.2 and 11.5 do not include Rubin.

| Device | Memory holding the weights | Bandwidth | Chips to hold 16 GB | Bound on tokens/s per user |
|---|---|---|---|---|
| NVIDIA Rubin (product page, November 2026) | HBM4 | 19.2 TB/s | 1 | about 1,200 |
| NVIDIA Rubin (announcement, January 2026) | HBM4 | 22 TB/s | 1 | about 1,375 |

### Measured per-user speed of SRAM-based services

Independent measurements show what these designs achieve on larger models. Artificial Analysis, which benchmarks API providers with the same prompts, measured Cerebras serving gpt-oss-120B at about 2,700 output tokens per second with a time to first token of 280 ms in August 2025 [34]. In March 2026 it measured a pre-release Groq 3 LPX endpoint at 3,431 tokens per second on Gemma 4 31B with a 100K-token context, against 870 for the fastest public endpoint; the endpoint was private and was measured at a concurrency of one, so the comparison is indicative only [35]. These figures change from month to month as providers update their software, so the measurement date matters as much as the number.

### TensorFlow status (March 2026)

TensorFlow is in maintenance: with release 2.21 in March 2026, Google limited it to security, bug and dependency fixes and recommended Keras 3, JAX or PyTorch for new generative AI work [38], and its newest TPUs do not support it [23].

### TorchTPU status and roadmap (October 2026)

TorchTPU was announced by Google and Meta in October 2026 [41]. Its authors describe dynamic=False as a limitation they expect to remove. Google's documentation states that TorchTPU replaces PyTorch/XLA [42].

### Workaround: JAX and KerasHub in the Colab TPU image (November 2026)

The JAX path failed three times at import with a duplicate registration of an XLA flag, because the runtime image already contained TensorFlow, and installing KerasHub loaded a second copy of XLA into the same process. With TensorFlow removed and KerasHub installed without its dependencies, the same code ran; the version record of step 1 is what exposes such a clash. The case study ran this configuration as "jax.jit, TensorFlow removed from the image". Run records: the three failures are noted in [`results/colab_tpu_v6e1_2026-11-15.json`](results/colab_tpu_v6e1_2026-11-15.json) (TPU v6e-1, 15 November 2026), and the run with TensorFlow uninstalled and keras-hub installed with `--no-deps` is [`results/colab_tpu_v5e1_jax_2026-11-16.json`](results/colab_tpu_v5e1_jax_2026-11-16.json) (TPU v5e-1, 16 November 2026).

### Sample size of the JAX precision comparison

The JAX comparison used a smaller sample than the TorchTPU one, the last 32 prompt positions of 8 prompts against 32 generated positions of all 22, so the two KL values are comparable in size but not to the second digit. Results: [`results/colab_tpu_v5e1_jax_2026-11-16.json`](results/colab_tpu_v5e1_jax_2026-11-16.json) (JAX, key `v5e-1_jax_precision_vs_fp32_highest`) and [`results/colab_tpu_v6e1_2026-11-15.json`](results/colab_tpu_v6e1_2026-11-15.json) (TorchTPU, key `parity_vs_cpu_fp32`).
