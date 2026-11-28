# Chapter 12 online resources

The Mature ML System

This document collects the online resources that the print chapter refers to: the answers to the final quiz and the references cited in the chapter, numbered as in the text. The tables and notes moved out of the print chapter are in the [chapter README](README.md#online-resources).

## Final quiz answers

The same answers are in [`knowledge_check_solutions.md`](knowledge_check_solutions.md).

The quiz covers the whole book. Questions 1 to 10 name the chapter they come from; questions 11 to 15 are
calculations from Chapter 12. Section names refer to the chapter named in each answer.

### Question 1

A pipeline spends 30% of its time in retrieval and 70% in generation. What is the largest end-to-end speedup that a
tenfold faster retriever can give, and which law states the limit? (Chapters 1 and 2)

**Answer.** 1 / (0.7 + 0.3 / 10) = 1 / 0.73 ≈ 1.37×, by Amdahl's Law. Even an infinitely fast retriever gives at most
1 / 0.7 ≈ 1.43×.

**Explanation.** The speedup of the whole is limited by the share of time that the change does not touch (Chapter 2,
*Amdahl's Law*); Chapter 3 applies the same bound to the non-decode stages (Table 3.11).

### Question 2

Why does full fine-tuning with Adam in mixed precision need about 16 bytes per parameter, and what does that imply
for an 8B model on one 80 GB GPU? (Chapters 3 and 10)

**Answer.** 2 bytes for the 16-bit weights, 2 for their gradients, 4 for an FP32 master copy and 8 for the two FP32
Adam moments. An 8B model needs about 128 GB of model state before activations, more than one 80 GB GPU holds.

**Explanation.** Chapter 3, Table 3.4 (Llama 3 8B: 16 + 16 + 96 GB, and about 174 GB with activations and
FlashAttention); Chapter 10, *Memory of full fine-tuning*.

### Question 3

What is a graph break in torch.compile, and why can it reduce the speedup of compilation? (Chapter 4)

**Answer.** When Dynamo reaches Python it cannot trace into a tensor graph, it compiles the part it captured, runs
the untraceable code as residual bytecode in eager mode and resumes compiling afterward. Each break ends a compiled
region, so fusion stops there and per-kernel launch overhead returns; many small breaks leave a model that is
compiled and still pays most of the eager-mode cost.

**Explanation.** Chapter 4, *TorchDynamo: capturing a graph from ordinary Python* and Principle #8.

### Question 4

A decode step at batch 1 reads 16 GB of weights on a GPU with 3.35 TB/s of memory bandwidth and 989 TFLOP/s of BF16
compute. Is it compute-bound or memory-bound, and what is the lower bound on its duration? (Chapter 5)

**Answer.** Memory-bound: the step performs about 1 FLOP per byte, and the ridge point is 989 / 3.35 ≈ 295 FLOP/byte.
The lower bound is 16 GB / 3.35 TB/s ≈ 4.8 ms.

**Explanation.** Chapter 5, *Dense linear layers: decode and prefill* and *Performing Roofline analysis*.

### Question 5

What does tiling a matrix multiplication in Triton change about the traffic between global memory and on-chip
memory? (Chapter 6)

**Answer.** Each program instance loads a BLOCK_SIZE_M × BLOCK_SIZE_K tile of A and a BLOCK_SIZE_K × BLOCK_SIZE_N tile
of B from global memory once and uses them for BLOCK_SIZE_M × BLOCK_SIZE_N × BLOCK_SIZE_K multiply-accumulates on
chip. Each loaded byte is reused across more arithmetic as the tiles grow, so global-memory traffic falls relative to
the work and the arithmetic intensity rises toward the compute roof.

**Explanation.** Chapter 6, *Tiled Matrix Multiplication*.

### Question 6

In FSDP, which collective operations run in the forward and backward passes, and how does their volume compare with
DDP's all-reduce? (Chapter 7)

**Answer.** The forward pass all-gathers each layer's parameters before using them; the backward pass all-gathers
them again and reduce-scatters the gradients. Each moves about N bytes per GPU per step, about 3N in total against
2N for DDP's all-reduce, 50% more.

**Explanation.** Chapter 7, *What the sharding costs in communication*, Figure 7.12.

### Question 7

Why does the KV cache, and not the weights, limit the number of concurrent requests at long context? How does paged
attention change the number that fit? (Chapter 8)

**Answer.** The weights occupy memory once per GPU, while the KV cache occupies memory for every token of every
running request, so at long context the cache fills the memory left after the weights. Paged attention allocates the
cache in fixed-size blocks on demand, so a request wastes at most one partial block; earlier contiguous allocators
used only 20.4% to 38.2% of the cache memory for token states, and vLLM's throughput rose 2× to 4× over them at the
same latency.

**Explanation.** Chapter 8, Principles #26 and #27, *Memory waste in contiguous allocation* and *PagedAttention and
block tables*.

### Question 8

An FP8 checkpoint runs on an A100 and an H100. On which GPU does it speed up prefill, and why? (Chapter 9)

**Answer.** On the H100, whose Tensor Cores run FP8 at twice the BF16 rate (1,979 against 989 dense TFLOP/s). The
A100 has no FP8 arithmetic and converts the weights to 16 bits inside the kernel, so prefill runs at the 16-bit rate.

**Explanation.** Chapter 9, *Formats for quantized inference* and *Format selection by hardware*.

### Question 9

When should new knowledge be added to the retrieval index, and when should the model be fine-tuned? (Chapter 10)

**Answer.** New facts go into the retrieval index; fine-tuning is for new behaviors: how the model uses its context,
formats its output or decides to abstain.

**Explanation.** Chapter 10, Principle #41: updating an index takes minutes and leaves the model unchanged, and it
outperformed fine-tuning for injecting new facts in the cited studies and in the case study's knowledge probe.

### Question 10

Why does a dataflow accelerator with weights in on-chip SRAM reach higher single-user decode speed than an HBM-based
GPU, and what does it need more of to serve a 70B model? (Chapter 11)

**Answer.** Batch-1 decode is bound by the bandwidth of the memory that holds the weights, and SRAM bandwidth is far
higher: about 150 TB/s per Groq LP30 against 3.35 TB/s of HBM on an H100. SRAM capacity per chip is small (about
500 MB), so a 70B model needs many more chips: about 140 for the FP8 weights alone, before any KV cache.

**Explanation.** Chapter 11, *The Roofline model for SRAM-resident designs* and *What limits dataflow designs*.

### Question 11

Estimate the KV cache transfer time for a 2,048-token prompt of Llama-3.1-8B (128 KiB per token in BF16) over a
200 Gb/s link.

**Answer.** 2,048 × 128 KiB = 256 MiB, 268 MB. A 200 Gb/s link carries 25 GB/s, so the transfer takes about
268 MB / 25 GB/s ≈ 11 ms.

**Explanation.** The same calculation in *Disaggregated prefill and decode* gives 1.34 GB and about 27 ms for a
4,096-token Llama-3.1-70B prompt over 400 Gb/s; engines hide most of it by sending each layer's cache when that layer
finishes.

### Question 12

One GPU completes 0.4 tasks per second within the objectives, and the service runs at 50% of that rate on average.
How many GPU-seconds does each task use, and what is the cost per task at a price of P per GPU-hour?

**Answer.** GPUs / (goodput × utilization) = 1 / (0.4 × 0.5) = 5.0 GPU-seconds per task (2.5 at full utilization).
The cost per task is P × 1 / (0.4 × 0.5 × 3,600) = P / 720.

**Explanation.** *Cost accounting: from GPU-hours to cost per task*, Code 12.1: the cost per task rises in inverse
proportion to utilization (Principle #47).

### Question 13

Compute the KV bytes per token in BF16 for a model with 32 layers and head dimension 128 with 32, 8 and 1 KV heads.

**Answer.** 2 × 32 × KV heads × 128 × 2 bytes = 16,384 bytes per KV head: 524,288 bytes (512 KiB) with 32 KV heads,
131,072 bytes (128 KiB) with 8, and 16,384 bytes (16 KiB) with 1.

**Explanation.** These are the MHA, GQA and MQA designs of *Architecture choices that set inference cost*; Chapter 8
gives 512 KB for Llama-2-7B and 128 KB for Llama-3.1-8B.

### Question 14

An agent's calls are spread round-robin over four replicas, each with its own prefix cache. One replica reaches a
prefix hit rate of 71%, and 48% of the prompt tokens are a shared system prompt and tool prefix. Ignoring eviction,
what hit rate does round-robin give, and what policy raises it?

**Answer.** Only the part outside the shared prefix follows the 1/N prediction: s + (71 − s) / N =
48 + 23 / 4 ≈ 54%. Prefix-aware (cache-aware) routing, which sends each call to the replica that holds the longest
prefix of its prompt, raises it.

**Explanation.** In experiment E8 with four replicas at 8 tasks per second, round-robin, session-affinity and
prefix-aware routing reached hit rates of 53%, 71% and 83% (*Case Study: The Mature Multimodal, Agentic RAG System*,
*Cache-aware routing*, Principle #49).

### Question 15

A workload has 2,000-token prompts and 2,000-token outputs at 10 requests per second. Using the method of Code 12.3,
explain whether you expect it to need more prefill or more decode capacity, and what would make disaggregation worth
its cost.

**Answer.** More decode capacity. With Code 12.3's assumptions (Llama-3.1-8B in BF16 on H100 GPUs, TPOT objective
50 ms), prefill needs 10 × 2,000 × 16.06 GFLOP / (0.5 × 989 TFLOP/s) ≈ 0.65 GPUs. Decode at a mean context of
3,000 tokens fits a batch of 146 in the KV cache, with a Roofline step of about 22 ms, so it needs
10 × 2,000 × 0.022 s / 146 ≈ 3.0 GPUs, about 4.6 times the prefill capacity. Disaggregation pays when strict
objectives for both phases cannot be met on shared GPUs, when the prefill-to-decode ratio is far from what one pool
provides, or when the two phases can run on different, cheaper hardware; for a single node with a mixed workload,
chunked prefill on shared GPUs is usually the better first step.

**Explanation.** *Disaggregated prefill and decode*, Code 12.3 and Figure 12.4 (the reasoning workload, with long
outputs, needs twenty times more decode than prefill); Principle #48.

## References

The 72 references cited in Chapter 12, numbered as in the text.

**[1]** vLLM Project, "v0.33.0" release notes, 1 November 2026. <https://github.com/vllm-project/vllm/releases/tag/v0.33.0>

**[2]** vLLM Project, "Production Metrics," vLLM documentation. Accessed 7 November 2026. <https://docs.vllm.ai/en/latest/usage/metrics/>

**[3]** OpenTelemetry, "Semantic Conventions for Generative AI Systems" (status: Development). Accessed 7 November 2026. <https://opentelemetry.io/docs/specs/semconv/gen-ai/>

**[4]** GetDeploying, "GPU Price Trends," weekly median on-demand prices across providers. Accessed 7 November 2026. <https://getdeploying.com/gpu-price-trends>

**[5]** Kubernetes SIG Network, "Introduction - Kubernetes Gateway API Inference Extension," documentation. Accessed 25 October 2026. <https://gateway-api-inference-extension.sigs.k8s.io/>

**[6]** llm-d Project, "llm-d v1.0: State-of-the-Art Inference, Ready for Day 2," llm-d blog, October 2026. <https://llm-d.ai/blog/llm-d-v1.0-state-of-the-art-inference-ready-for-day-2>

**[7]** NVIDIA, "Release Notes," NVIDIA Dynamo documentation (v1.6.0, 22 October 2026). Accessed 7 November 2026. <https://docs.dynamo.nvidia.com/dynamo/dev/reference/releases>

**[8]** KServe Project, "Announcing KServe v0.18," KServe blog, 29 April 2026. <https://kserve.github.io/website/blog/kserve-0.18-release>

**[9]** NVIDIA, "NIXL: NVIDIA Inference Xfer Library," GitHub repository. Accessed 25 October 2026. <https://github.com/ai-dynamo/nixl>

**[10]** Cloud Native Computing Foundation, "Kubernetes Established as the De Facto ‘Operating System’ for AI as Production Use Hits 82% in 2025 CNCF Annual Cloud Native Survey," press release, 20 January 2026. <https://www.cncf.io/announcements/2026/01/20/kubernetes-established-as-the-de-facto-operating-system-for-ai-as-production-use-hits-82-in-2025-cncf-annual-cloud-native-survey/>

**[11]** NVIDIA, "Router Design," NVIDIA Dynamo documentation, version 1.0.1. Accessed 25 October 2026. <https://docs.nvidia.com/dynamo/v1.0.1/design-docs/component-design/router-design>

**[12]** Ayoub, M., et al., "KV-Cache Wins You Can See: From Prefix Caching in vLLM to Distributed Scheduling with llm-d," llm-d blog, September 2025. <https://llm-d.ai/blog/kvcache-wins-you-can-see>

**[13]** Agrawal, A., et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve," OSDI 2024. <https://arxiv.org/abs/2403.02310>

**[14]** Zhong, Y., et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving," OSDI 2024. <https://arxiv.org/abs/2401.09670>

**[15]** Patel, P., et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting," ISCA 2024. <https://arxiv.org/abs/2311.18677>

**[16]** Qin, R., et al., "Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot," FAST 2025. <https://www.usenix.org/conference/fast25/presentation/qin>

**[17]** vLLM Project, vLLM blog post on encoder-prefill-decode disaggregation, 15 December 2025. <https://vllm.ai/blog/vllm-epd>

**[18]** Liu, Y., et al., "LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference," 2025. <https://arxiv.org/abs/2510.09665>

**[19]** DeepSeek-AI, "DeepEP: An Efficient Expert-Parallel Communication Library," GitHub repository (V1 performance in docs/legacy.md). Accessed 25 October 2026. <https://github.com/deepseek-ai/DeepEP/blob/main/docs/legacy.md>

**[20]** LMSYS, "Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism on 96 H100 GPUs," May 2025. <https://lmsys.org/blog/2025-05-05-large-scale-ep/>

**[21]** DeepSeek-AI, "Day 6: One More Thing, DeepSeek-V3/R1 Inference System Overview," February 2025. <https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md>

**[22]** vLLM Project, vLLM blog post on large-scale DeepSeek serving with wide expert parallelism, 17 December 2025. <https://vllm.ai/blog/2025-12-17-large-scale-serving>

**[23]** Zhu, K., et al., "TraceLab: Characterizing Coding Agent Workloads for LLM Serving," 2026. <https://arxiv.org/abs/2606.30560>

**[24]** Yuan, Z., et al., "Agentic AI Workload Characteristics," 2026. <https://arxiv.org/abs/2605.26297>

**[25]** Abhyankar, R., et al., "InferCept: Efficient Intercept Support for Augmented Large Language Model Inference," ICML 2024. <https://arxiv.org/abs/2402.01869>

**[26]** Li, H., et al., "Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live," 2025. <https://arxiv.org/abs/2511.02230>

**[27]** Pan, Z., et al., "KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows," NeurIPS 2025. <https://papers.neurips.cc/paper_files/paper/2025/hash/b7971d31a7d5eb0f1eed2f8f6f368195-Abstract-Conference.html>

**[28]** Luo, M., et al., "Autellix: An Efficient Serving Engine for LLM Agents as General Programs," 2025. <https://arxiv.org/abs/2502.13965>

**[29]** Oliaro, G., et al., "SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications," NeurIPS 2025. <https://arxiv.org/abs/2411.04975>

**[30]** NVIDIA, "Fault Tolerance," NVIDIA Dynamo documentation, version 1.0.0. <https://docs.nvidia.com/dynamo/v1.0.0/user-guides/fault-tolerance>

**[31]** vLLM Project, vLLM blog post on sleep mode, 26 October 2025. <https://vllm.ai/blog/2025-10-26-sleep-mode>

**[32]** Miao, X., et al., "SpotServe: Serving Generative Large Language Models on Preemptible Instances," ASPLOS 2024. <https://arxiv.org/abs/2311.15566>

**[33]** Sculley, D., et al., "Hidden Technical Debt in Machine Learning Systems," NeurIPS 2015. <https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems>

**[34]** He, H., and Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference," Thinking Machines Lab: Connectionism, September 2025. <https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/>

**[35]** vLLM Project, "Batch Invariance," vLLM documentation. Accessed 7 November 2026. <https://docs.vllm.ai/en/latest/features/batch_invariance/>

**[36]** vLLM Project, "Performance Dashboard," vLLM documentation. Accessed 7 November 2026. <https://docs.vllm.ai/en/latest/benchmarking/dashboard/>

**[37]** SemiAnalysis, "InferenceX: Open Continuous Inference Benchmark." Accessed 7 November 2026. <https://inferencex.semianalysis.com/about>

**[38]** MLCommons, "MLPerf Inference v6.1 Results," September 2026; summary in StorageReview, 16 September 2026. <https://www.storagereview.com/news/mlperf-inference-v6-1-5-7x-per-accelerator-gains-a-512-gpu-run-and-vera-rubins-first-peer-reviewed-numbers>

**[39]** Zheng, L., et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," NeurIPS 2023 Datasets and Benchmarks. <https://arxiv.org/abs/2306.05685>

**[40]** Beyer, B., Murphy, N. R., Rensin, D. K., Kawahara, K., and Thorne, S. (eds.), "Canarying Releases," in The Site Reliability Workbook, O'Reilly, 2018. <https://sre.google/workbook/canarying-releases/>

**[41]** PyTorch, "RELEASE.md: Release Compatibility Matrix and Cadence." Accessed 7 November 2026. <https://github.com/pytorch/pytorch/blob/main/RELEASE.md>

**[42]** PyTorch, "PyTorch 2.15.0 General Availability," PyTorch Developer Discussions, 28 October 2026. <https://dev-discuss.pytorch.org/t/pytorch-2-15-0-general-availability/3470>

**[43]** Gemma Team, "Gemma 3 Technical Report," 2025. <https://arxiv.org/abs/2503.19786>

**[44]** Lieber, O., et al., "Jamba: A Hybrid Transformer-Mamba Language Model," 2024. <https://arxiv.org/abs/2403.19887>

**[45]** Dao, T., and Gu, A., "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality," ICML 2024. <https://arxiv.org/abs/2405.21060>

**[46]** DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model," 2024. <https://arxiv.org/abs/2405.04434>

**[47]** NVIDIA, "Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models," 2025. <https://arxiv.org/abs/2504.03624>

**[48]** Bercovich, A., et al., "Puzzle: Distillation-Based NAS for Inference-Optimized LLMs," 2024. <https://arxiv.org/abs/2411.19146>

**[49]** Wu, B., et al., "FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search," CVPR 2019. <https://arxiv.org/abs/1812.03443>

**[50]** Cai, H., Zhu, L., and Han, S., "ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware," ICLR 2019. <https://arxiv.org/abs/1812.00332>

**[51]** Cai, H., Gan, C., Wang, T., Zhang, Z., and Han, S., "Once-for-All: Train One Network and Specialize It for Efficient Deployment," ICLR 2020. <https://arxiv.org/abs/1908.09791>

**[52]** Wang, H., et al., "HAT: Hardware-Aware Transformers for Efficient Natural Language Processing," ACL 2020. <https://arxiv.org/abs/2005.14187>

**[53]** Bercovich, A., et al., "Llama-Nemotron: Efficient Reasoning Models," 2025. <https://arxiv.org/abs/2505.00949>

**[54]** Cai, R., et al., "Flextron: Many-in-One Flexible Large Language Model," ICML 2024. <https://cairuisi.github.io/Flextron>

**[55]** Epoch AI, "Output Length of Language Models," data insight, April 2025 (data updated November 2025). <https://epoch.ai/data-insights/output-length>

**[56]** Cottier, B., et al., "LLM Inference Prices Have Fallen Rapidly but Unequally Across Tasks," Epoch AI data insight, March 2025. <https://epoch.ai/data-insights/ai-inference-price-trends>

**[57]** International Energy Agency, "Energy and AI," April 2025. <https://www.iea.org/reports/energy-and-ai>

**[58]** International Energy Agency, "Data Centre Electricity Use Surged in 2025," news release, April 2026. <https://www.iea.org/news/data-centre-electricity-use-surged-in-2025-even-with-tightening-bottlenecks-driving-a-scramble-for-solutions>

**[59]** Shehabi, A., et al., "2024 United States Data Center Energy Usage Report," Lawrence Berkeley National Laboratory, December 2024. <https://eta.lbl.gov/publications/2024-lbnl-data-center-energy-usage-report>

**[60]** Elsworth, C., et al., "Measuring the Environmental Impact of Delivering AI at Google Scale," 2025. <https://arxiv.org/abs/2508.15734>

**[61]** NVIDIA, "Pretraining Large Language Models with NVFP4," 2025. <https://arxiv.org/abs/2509.25149>

**[62]** Ambrogio, S., et al., "An Analog-AI Chip for Energy-Efficient Speech Recognition and Transcription," Nature 620, 768–775, 2023. <https://www.nature.com/articles/s41586-023-06337-5>

**[63]** Büchel, J., et al., "Analog Foundation Models," NeurIPS 2025. <https://proceedings.neurips.cc/paper_files/paper/2025/hash/580baa016b012f8d3ea27b9ac758cee1-Abstract-Conference.html>

**[64]** Ahmed, S. R., et al., "Universal Photonic Artificial Intelligence Acceleration," Nature 640, 368–374, 2025. <https://doi.org/10.1038/s41586-025-08854-x>

**[65]** Chen, J., et al., "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation," Findings of ACL 2024. <https://arxiv.org/abs/2402.03216>

**[66]** TomoroAI, "tomoro-colqwen3-embed-4b" and its AWQ variant, model cards (Apache 2.0). Accessed 7 November 2026. <https://huggingface.co/TomoroAI/tomoro-ai-colqwen3-embed-4b-awq>

**[67]** Faysse, M., et al., "ColPali: Efficient Document Retrieval with Vision Language Models," ICLR 2025. <https://arxiv.org/abs/2407.01449>

**[68]** Loison, A., et al., "ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios," 2026. <https://arxiv.org/abs/2601.08620>

**[69]** Cormack, G. V., Clarke, C. L. A., and Büttcher, S., "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods," SIGIR 2009. <https://doi.org/10.1145/1571941.1572114>

**[70]** Wang, Q., et al., "ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents," 2025. <https://arxiv.org/abs/2502.18017>

**[71]** Qwen Team, "Qwen3-VL Technical Report," 2025, and model card Qwen/Qwen3-VL-8B-Instruct (Apache 2.0). <https://arxiv.org/abs/2511.21631>

**[72]** vLLM Project, "Disaggregated Prefilling (experimental)," vLLM documentation, v0.33.0, accessed 27 November 2026. <https://docs.vllm.ai/en/v0.33.0/features/disagg_prefill/>
