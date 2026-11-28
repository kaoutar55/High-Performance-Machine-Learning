# Chapter 11 online resources

Navigating the Accelerator and Compiler Landscape

This document collects the online resources that the print chapter refers to: the answers to the knowledge check and the references cited in the chapter, numbered as in the text. The tables and notes moved out of the print chapter are in the [chapter README](README.md#online-resources).

## Knowledge check answers

The same answers are in [`knowledge_check_solutions.md`](knowledge_check_solutions.md).

Section names refer to the chapter.

### Question 1

An accelerator offers 2,300 TFLOP/s in BF16 and 7.4 TB/s of HBM bandwidth. Compute its BF16 ridge point and the
batch-size-1 decode bound for a model with 16 GB of BF16 weights. Which property would you change to double the
tokens per second at batch size 1?

**Answer.** Ridge point: 2,300 × 10¹² / 7.4 × 10¹² ≈ 311 FLOP/B. Decode bound: 16 GB / 7.4 TB/s ≈ 2.2 ms per token,
about 460 tokens/s. Batch-1 decode at about 1 FLOP/B is bound by memory bandwidth, so doubling the bandwidth (or
halving the bytes read per token) doubles the bound; more compute changes nothing.

**Explanation.** The ridge point is peak compute over bandwidth for the format in use, and decode at small batch is
bound by the bandwidth of the memory that holds the weights (*Five properties that describe an accelerator*,
Principle #42).

### Question 2

Why does a TPU program pad its inputs to a few fixed lengths, and what does the same technique achieve in a GPU
serving engine?

**Answer.** XLA compiles for static shapes, so every distinct prompt length would compile a new program; padding to
a small set of bucket sizes bounds the number of compiled programs. Serving engines use the same technique to bound
the number of CUDA graphs they capture, padding each step to the nearest captured batch size.

**Explanation.** *PyTorch beyond NVIDIA GPUs* (TorchTPU compiles with `dynamic=False`) and *What transfers between
stacks*; Chapter 8 describes the CUDA-graph padding.

### Question 3

An SRAM-based chip holds about 500 MB. How many chips does a 70B-parameter model in FP8 need for its weights alone,
and what else must fit in fast memory during long-context decode?

**Answer.** 70 × 10⁹ bytes / 500 MB ≈ 140 chips. The KV cache of every running request must also fit: at 128 KiB per
token for Llama-3.1-8B in BF16, a single 100K-token request needs about 13 GB, and it competes with the weights for
SRAM.

**Explanation.** *What limits dataflow designs*, Figure 11.6; one survey estimated about 576 first-generation Groq
chips for a 70B model.

### Question 4

The same prompt sent at temperature 0 returns two different completions at different times of day. Name the most
likely cause, and describe how you would confirm it.

**Answer.** Kernels that are not batch-invariant: a sequence's result depends on the batch it is processed with, and
under load the batch changes with every step. To confirm, send the same prompt many times while a background load
keeps different numbers of requests running, count the distinct completions and the first divergent token, and
repeat with batch-invariant kernels (`VLLM_BATCH_INVARIANT=1` in vLLM); the divergence should disappear.

**Explanation.** In the chapter's experiment (Llama-3.1-8B, vLLM 0.33.0), 100 identical requests gave one
completion alone on the GPU and up to 4 (A100) and 12 (H100) distinct completions with 64 requests running, and a
single completion in batch-invariant mode, at a throughput cost (*Determinism constraints*, *Measuring
nondeterminism on the case-study model*, Principle #44).

### Question 5

A model ported to a new accelerator matches the reference on every operator test but scores two points lower on its
task evaluation. List the steps you would take to find the cause.

**Answer.** Record and compare the full environment (drivers, framework, compiler, serving engine, model revision,
tokenizer, container digest, precision settings). Run a layer-level comparison with hooks that save every decoder
layer's output and look for a jump at one layer. Run model-level checks: KL divergence and top-1 agreement of the
next-token distributions against an FP32 reference, then greedy generations and task accuracy with paired error
bars. Check operator coverage and fallbacks, check precision defaults and the serving-stack configuration, and bisect
to a minimal reproduction.

**Explanation.** These are the steps of *Debugging Methodology for New Hardware* (Figure 11.9). Model-level checks
catch errors that operator tests miss: providers serving the same gpt-oss-120B weights scored between 80.0% and
93.3% on AIME 2025 because of their serving stacks (Principle #46).

### Question 6

Your workload is interactive decode for a 70B model with 32K-token contexts. Which properties from Figure 11.1 limit
it, and which evidence would you ask each vendor for?

**Answer.** Memory bandwidth of the memory that holds the weights and KV cache sets the per-token time; memory
capacity sets how many chips the weights and the 32K-token caches need; interconnect bandwidth matters once the
model is split across chips; and the software stack sets how close the workload comes to the bandwidth roof. Ask for
per-user tokens per second and TTFT, and capacity and TPOT, at 32K tokens of context, measured with the serving
stack you will deploy.

**Explanation.** Table 11.6 lists bandwidth for interactive decode and HBM capacity and scale-up bandwidth for long
context, with this evidence (*A decision table*, Principles #42 and #43).

### Question 7

An FP8 checkpoint runs on a GPU whose FP8 rate is twice its BF16 rate, and on an accelerator whose FP8 rate equals
its BF16 rate. On which phase would you expect FP8 to help on each, and why?

**Answer.** On the GPU, FP8 helps both phases: prefill, which is compute-bound, runs at twice the rate, and decode,
which is bandwidth-bound, reads half the weight bytes. On the accelerator (for example Gaudi 3, which runs FP8 at its
BF16 rate), FP8 helps decode through the halved bytes and gives prefill no compute gain.

**Explanation.** Compare accelerators on the roof that bounds each phase (Principle #42, *Number formats as a
hardware feature*). On the H100, FP8 weights cut Llama-3.1-8B's time per output token by 1.5× against a bound ratio
of 1.8, because the embedding and output head stay in BF16 (Table 11.10).

### Question 8

A JAX training step slows down after you change the sharding of one weight, and the Python code shows no new
communication. Where would you look, and what would you expect to find?

**Answer.** In the compiled program (`jit(...).lower(...).compile().as_text()`) or a profile: in JAX the shardings
propagate and the compiler inserts the collectives, so you would expect new all-gathers or reduce-scatters, or a
resharding of the changed weight, that the Python code does not show.

**Explanation.** *Sharding in JAX and PyTorch*; the communication costs of Chapter 7 apply unchanged.

## References

The 67 references cited in Chapter 11, numbered as in the text.

**[1]** Prabhakar, R., et al., "SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts," MICRO 2024. <https://arxiv.org/abs/2405.07518>

**[2]** NVIDIA, "NVIDIA H100 Tensor Core GPU," datasheet. <https://www.nvidia.com/en-us/data-center/h100/>

**[3]** Horowitz, M., "Computing's Energy Problem (and What We Can Do About It)," ISSCC 2014. <https://doi.org/10.1109/ISSCC.2014.6757323>

**[4]** Micikevicius, P., et al., "Mixed Precision Training," ICLR 2018. <https://arxiv.org/abs/1710.03740>

**[5]** Kalamkar, D., et al., "A Study of BFLOAT16 for Deep Learning Training," 2019. <https://arxiv.org/abs/1905.12322>

**[6]** NVIDIA, "Accelerating AI Training with NVIDIA TF32 Tensor Cores," NVIDIA Technical Blog, 2021. <https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/>

**[7]** JAX authors, "Matmul precision," JAX documentation. <https://docs.jax.dev/en/latest/201/precision.html>

**[8]** Jouppi, N. P., et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit," ISCA 2017. <https://arxiv.org/abs/1704.04760>

**[9]** NVIDIA, "NVIDIA T4 Tensor Core GPU," datasheet. <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf>

**[10]** Micikevicius, P., et al., "FP8 Formats for Deep Learning," 2022. <https://arxiv.org/abs/2209.05433>

**[11]** DeepSeek-AI, "DeepSeek-V3 Technical Report," 2024. <https://arxiv.org/abs/2412.19437>

**[12]** Open Compute Project, "AMD, Arm, Intel, Meta, Microsoft, NVIDIA, and Qualcomm Standardize Next-Generation Narrow Precision Data Formats for AI," 2023 (OCP Microscaling Formats MX v1.0 Specification). <https://www.opencompute.org/blog/amd-arm-intel-meta-microsoft-nvidia-and-qualcomm-standardize-next-generation-narrow-precision-data-formats-for-ai>

**[13]** Rouhani, B. D., et al., "Microscaling Data Formats for Deep Learning," 2023. <https://arxiv.org/abs/2310.10537>

**[14]** NVIDIA, "Introducing NVFP4 for Efficient and Accurate Low-Precision Inference," NVIDIA Technical Blog, 2025. <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference>

**[15]** AWS, "Trainium3 architecture," AWS Neuron documentation, and "Amazon EC2 Trn3 instances." <https://aws.amazon.com/ec2/instance-types/trn3/>

**[16]** NVIDIA, "NVIDIA A100 Tensor Core GPU," datasheet. <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf>

**[17]** AMD, "AMD Instinct MI300X Accelerator," data sheet. <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf>

**[18]** Intel, "Intel Gaudi 3 AI Accelerator," white paper 817486, 2024. <https://www.intel.com/content/www/us/en/content-details/817486/intel-gaudi-3-ai-accelerator-white-paper.html>

**[19]** NVIDIA, "NVIDIA HGX Platform," product page with HGX B200 and HGX B300 specifications (sparse and dense Tensor Core rates). <https://www.nvidia.com/en-us/data-center/hgx/>

**[20]** AMD, "AMD Instinct MI355X GPU," product brief. <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf>

**[21]** Google Cloud, "TPU v5e," Cloud TPU documentation. <https://docs.cloud.google.com/tpu/docs/v5e>

**[22]** Google Cloud, "TPU v6e," Cloud TPU documentation. <https://docs.cloud.google.com/tpu/docs/v6e>

**[23]** Google Cloud, "TPU7x (Ironwood)," Cloud TPU documentation. <https://docs.cloud.google.com/tpu/docs/tpu7x>

**[24]** The Register, "Cerebras' wafer-scale AI bet delivers blockbuster IPO," May 15, 2026. <https://www.theregister.com/ai-ml/2026/05/15/cerebras-wafer-scale-ai-bet-delivers-blockbuster-ipo/5240821>

**[25]** The Register, "Nvidia LPX deep dive," March 19, 2026. <https://www.theregister.com/2026/03/19/nvidia_lpx_deep_dive/>

**[26]** IBM, "IBM Introduces the Spyre Accelerator for Commercial Availability," IBM Newsroom, October 7, 2025. <https://newsroom.ibm.com/2025-10-07-ibm-introduces-the-spyre-accelerator-for-commercial-availability>

**[27]** Dally, W. J., Turakhia, Y., and Han, S., "Domain-Specific Hardware Accelerators," Communications of the ACM 63(7), 2020. <https://doi.org/10.1145/3361682>

**[28]** Kung, H. T., "Why Systolic Architectures?," IEEE Computer 15(1), 1982. <https://doi.org/10.1109/MC.1982.1653825>

**[29]** AWS, "Trainium2 architecture," AWS Neuron documentation. <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html>

**[30]** Dennis, J. B., and Misunas, D. P., "A Preliminary Architecture for a Basic Data-Flow Processor," ISCA 1975. <https://doi.org/10.1145/642089.642111>

**[31]** Abts, D., Ross, J., et al., "Think Fast: A Tensor Streaming Processor (TSP) for Accelerating Deep Learning Workloads," ISCA 2020. <https://groq.com/wp-content/uploads/2024/02/2020-Isca.pdf>

**[32]** NVIDIA, "NVIDIA Vera Rubin NVL72," product page (preliminary specifications). <https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72>

**[33]** Apple Machine Learning Research, "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU," November 19, 2025. <https://machinelearning.apple.com/research/exploring-llms-mlx-m5>

**[34]** Cerebras, "OpenAI gpt-oss-120B runs fastest on Cerebras," August 6, 2025 (reporting Artificial Analysis measurements). <https://cerebras.ai/blog/openai-gpt-oss-120b-runs-fastest-on-cerebras>

**[35]** Tom's Hardware, "Nvidia presents Groq 3 LPX architecture and unveils its first third-party inference benchmark," March 2026. <https://tomshardware.com/tech-industry/semiconductors/nvidia-presents-groq-3-lpx-architecture-and-unveils-its-first-third-party-inference-benchmark>

**[36]** Patel, A., and Singh, R., "AI Hardware Accelerators for LLMs: Architectures and the Memory Wall," 2026. <https://arxiv.org/abs/2608.28048>

**[37]** JAX authors, "JAX changelog" (jax 0.12.0, October 2026) and "Distributed arrays and automatic parallelization." <https://docs.jax.dev/en/latest/changelog.html>

**[38]** Google, "What's new in TensorFlow 2.21," Google Developers Blog, March 2026. <https://developers.googleblog.com/en/whats-new-in-tensorflow-221/>

**[39]** PyTorch, "Facilitating New Backend Integration by PrivateUse1," PyTorch documentation. <https://docs.pytorch.org/tutorials/advanced/privateuseone.html>

**[40]** torch-spyre project, "torch-spyre: PyTorch support for the IBM Spyre accelerator," documentation. <https://torch-spyre.readthedocs.io/>

**[41]** PyTorch Team and Google, "Introducing TorchTPU," PyTorch Blog, October 2026. <https://pytorch.org/blog/introducing-torchtpu/>

**[42]** Google Cloud, "PyTorch on Cloud TPU," Cloud TPU documentation. <https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu>

**[43]** InfoQ, "Google and Meta Introduce TorchTPU," October 25, 2026. <https://infoq.com/news/2026/10/google-meta-torchtpu>

**[44]** vLLM Team, "vLLM TPU: A New Unified Backend Supporting PyTorch and JAX on TPU," vLLM Blog, October 16, 2025. <https://vllm.ai/blog/2025-10-16-vllm-tpu>

**[45]** SemiAnalysis, "MI300X vs H100 vs H200 Benchmark Part 1: Training – CUDA Moat Still Alive," December 22, 2024. <https://newsletter.semianalysis.com/p/mi300x-vs-h100-vs-h200-benchmark-part-1-training>

**[46]** StorageReview, "MLPerf Inference v6.1: 5.7x per-accelerator gains, a 512-GPU run and Vera Rubin's first peer-reviewed numbers," September 16, 2026. <https://www.storagereview.com/news/mlperf-inference-v6-1-5-7x-per-accelerator-gains-a-512-gpu-run-and-vera-rubins-first-peer-reviewed-numbers>

**[47]** SemiAnalysis, "InferenceX: DeepSeek-V4 Pro AgentX, MI355X vs B200," August 2026. <https://inferencex.semianalysis.com/blog/deepseek-v4-pro-agentx-mi355x-vs-b200-august>

**[48]** vLLM Team, "ROCm attention backends in vLLM," vLLM Blog, February 27, 2026. <https://vllm.ai/blog/2026-02-27-rocm-attention-backend>

**[49]** AMD, "AMD Instinct MI300 Series microarchitecture," ROCm documentation. <https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi300.html>

**[50]** He, H., and Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference," Connectionism, September 2025. <https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/>

**[51]** vLLM, "Batch invariance," vLLM documentation. <https://docs.vllm.ai/en/latest/features/batch_invariance/>

**[52]** PyTorch, "Reproducibility," PyTorch documentation (torch.use_deterministic_algorithms and CUBLAS_WORKSPACE_CONFIG). <https://docs.pytorch.org/docs/stable/notes/randomness.html>

**[53]** ONNX Runtime, "Execution Providers," ONNX Runtime documentation (onnxruntime 1.32). <https://onnxruntime.ai/docs/execution-providers/>

**[54]** PyTorch, "torch.export-based ONNX Exporter," PyTorch documentation. <https://docs.pytorch.org/docs/stable/onnx_export.html>

**[55]** OpenXLA, "StableHLO specification." <https://openxla.org/stablehlo/spec>

**[56]** Lattner, C., et al., "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation," CGO 2021. <https://arxiv.org/abs/2002.11054>

**[57]** Google Open Source Blog, "PJRT: Simplifying ML Hardware and Framework Integration," 2023. <https://opensource.googleblog.com/2023/05/pjrt-simplifying-ml-hardware-and-framework-integration.html>

**[58]** NVIDIA, "TensorRT-LLM," documentation. <https://nvidia.github.io/TensorRT-LLM/>

**[59]** PyTorch, "AOTInductor: Ahead-Of-Time Compilation for torch.export-ed Models," PyTorch documentation. <https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html>

**[60]** PyTorch Team, "Introducing ExecuTorch 1.0," PyTorch Blog, October 2025. <https://pytorch.org/blog/introducing-executorch-1-0/>

**[61]** IREE project, "IREE: an MLIR-based compiler and runtime." <https://iree.dev/>

**[62]** Chen, T., et al., "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning," OSDI 2018. <https://arxiv.org/abs/1802.04799>

**[63]** Willison, S., "Inconsistent performance of gpt-oss-120b across providers," August 15, 2025 (reporting Artificial Analysis AIME25 measurements). <https://simonwillison.net/2025/Aug/15/inconsistent-performance/>

**[64]** OpenXLA, "HLO dumps," XLA documentation. <https://openxla.org/xla/hlo_dumps>

**[65]** Artificial Analysis, "gpt-oss-120B (high): API Provider Benchmarking," snapshot of October 26, 2026. <https://artificialanalysis.ai/models/gpt-oss-120b/providers>

**[66]** Gemma Team, Google DeepMind, "Gemma: Open Models Based on Gemini Research and Technology," 2024. <https://arxiv.org/abs/2403.08295>

**[67]** Keras team, "KerasHub: Gemma models," documentation. <https://keras.io/keras_hub/>
