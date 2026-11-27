# Chapter 12: The Mature ML System

**Status: case study run on 2026-11-19 to 2026-11-23 (H100).** Results are in `results/case_study.json`, the file the
chapter build reads. E9 (cost per task by GPU) ran the final configuration on an L40S and an A100 on Modal; the
optional llm-d run on Kubernetes (`k8s/`) was not run.

Companion code for Chapter 12: the mature multimodal, agentic RAG system. The case study serves Qwen3-VL-8B-Instruct
with vLLM 0.33.0 behind an agent that retrieves from a BGE-M3 + FAISS text index and a ColQwen3 page-image index over
the ViDoRe V3 corpora (computer science, finance and HR). A cumulative ladder of experiments (E0 to E7) adds one
serving optimization at a time on H100 GPUs and measures goodput, latency, accuracy and GPU-seconds per completed task.
E8 measures scale-out with three routing policies, E9 the choice of GPU, R0 retrieval quality, R1 run-to-run
variation and a vLLM version regression gate, and R2 batch invariance.

| File | Purpose |
|---|---|
| `build_corpus.py` | Download ViDoRe V3 CS, Finance-EN and HR at run time; render pages; license report |
| `indexing.py` | BGE-M3 + FAISS text index, ColQwen3 page index, retrieval service (Modal, or HTTP when `RETRIEVAL_URL` is set), R0 evaluation |
| `serve_vlm.py` | vLLM 0.33.0 server for Qwen3-VL-8B-Instruct on Modal, one configuration per experiment |
| `configs/experiments.yaml` | The cumulative experiments E0–E9, R1, R2, load points, objectives |
| `agent.py`, `tools.json`, `system_prompt.txt` | Agent controller, tools, OpenTelemetry-named spans |
| `router.py`, `routed_client.py` | Round-robin, session-affinity and prefix-aware routing (E8) |
| `loadgen_agent.py` | Open-loop multi-turn load generator |
| `evaluate.py`, `metrics.py` | Exact/numeric match, then a position-swapped judge; nDCG and recall |
| `cost.py` | Cost per completed task |
| `regress.py` | The regression gate |
| `smoke_test.py` | Checks vLLM 0.33.0 features before any experiment |
| `report.py` | Builds results/case_study.json for the chapter |
| `k8s/` | Optional llm-d 1.0 values for the Kubernetes run |

## Experimental setup

| Item | Value |
|---|---|
| Hardware | NVIDIA H100 80GB HBM3 (SXM, 700 W), driver 590.48.01, on a GPU cluster with 8 H100 GPUs per node (NVLink 4 through NVSwitch). One H100 for vLLM; E0-E6 run the retriever on a second H100 of the same node; E7 shares one GPU; E8 uses 2 or 4 GPUs of one node |
| Software | vLLM 0.33.0 (R1 baseline: 0.32.0), torch 2.14.0+cu130, transformers 5.20.2 (serving) / 5.23.0 (retriever), FlagEmbedding, faiss-cpu, auto-round 0.16.1 |
| Generator | Qwen/Qwen3-VL-8B-Instruct @0c351dd0 (E0-E4); Qwen/Qwen3-VL-8B-Instruct-FP8 @9cdc6310 (E5 onward) |
| Retrievers | BAAI/bge-m3 + FAISS inner product (text); TomoroAI/tomoro-colqwen3-embed-4b @13517a29 (FP16) and tomoro-ai-colqwen3-embed-4b-awq @309474f5 (4-bit, E7), MaxSim on the GPU |
| Judge | Qwen/Qwen3-32B-FP8 @aa55da1e on its own H100, thinking mode off; used only when exact or numeric matching fails; both answer orders must agree |
| Corpus | ViDoRe V3 computer science, finance (English) and HR at pinned revisions: 5,412 pages, 842 English questions, graded qrels |
| Load | open-loop Poisson task arrivals; rates 0.5, 1, 2, 4, 8, 16 tasks/s in ascending order, stopping one rate above goodput; 600 s per rate after 60 s warm-up; 3 repeats per rate (E8: 1) |
| Objective | task completion time <= 20 s; goodput = highest rate at which >= 90% of tasks met it in all 3 repeats; table rows are medians of the 3 repeats at goodput |
| Cost | GPU-seconds per task = GPUs / goodput, at utilization 1.0 (E0-E6: two H100s; E7 and E9: one GPU). Cost per task at a price per GPU-hour = GPU-seconds per task x price / 3,600 (`cost.py`); see [GPU-seconds and cost per task](#gpu-seconds-and-cost-per-task-for-the-case-study) |

Notes:

- Agent: the first model call of each task must be a tool call (`require_tool_first: true` in `configs/experiments.yaml`).
  Without it, Qwen3-VL-8B answered 46% of E2 tasks at step 0 without opening any document (accuracy 10.6% on those
  tasks). The tool list and the prompt are unchanged, so prefix reuse is not affected. E0 does not use the agent loop.
- The FP8 linear layers (E5 onward) JIT-compile FlashInfer kernels at start-up and need `nvcc` from a CUDA 13.0 toolkit.
- E7's page retriever is compiled with `torch.compile(dynamic=True)` and warmed up on 16 query lengths before serving;
  a static compile recompiled for each new query length and stalled requests.
- The 4-bit page model is loaded with the FP16 repository's model code, which supports transformers 5.x.
- The regression gate (R1) runs the full 842-question set at 4 tasks/s, 3 repeats per vLLM version.
- The judge runs with Qwen3's thinking mode off (`chat_template_kwargs: {enable_thinking: false}`); with thinking on,
  its 3-token reply never reached YES.
- Judge check: a second judge from another model family (google/gemma-3-27b-it) labelled 99 sampled E7 answers;
  agreement with the case study's verdicts 84.8% (Cohen's kappa 0.60), and 86.0% (kappa 0.72) on the
  answers that the first judge decided (`results/judge_agreement.json`). The author labelled the 15 answers where the
  two judges disagreed; with these labels, 89.9% of the case-study verdicts on the sample hold (92.0% of the "correct"
  verdicts and 83.3% of the "wrong" verdicts). Exact or numeric matching is the weaker part (87.8%, against 92.0% for
  the judge). The sample is half judged-correct and half judged-wrong, so these figures describe the verdicts, not task accuracy.

## Results

Retrieval (R0, 842 questions, nDCG@10): text 48.3, page image 69.5, hybrid (reciprocal rank
fusion) 60.3, page image in 4 bits 69.3. Recall@5: 40.7, 60.2, 52.0, 60.1.

### The cumulative ladder (H100, median of 3 repeats at goodput)

| ID | Change | Accuracy (%) | Task time p50 / p95 (s) | Steps per task | Prefix hit rate (%) | Images per task | Goodput (tasks/s) | GPUs | GPU-seconds per task |
|---|---|---|---|---|---|---|---|---|---|
| E0 | Single pass: hybrid retrieval, top-3 pages to the model | 56.2 | 4.6 / 11.7 | 1.0 | 5 | 3.0 | 1 | 2 | 2.0 |
| E1 | Agent loop (unstable prompt prefix) | 44.0 | 4.4 / 11.2 | 4.0 | 1 | 1.9 | 1 | 2 | 2.0 |
| E2 | Stable prompt prefix | 45.7 | 4.9 / 11.5 | 4.0 | 61 | 1.8 | 2 | 2 | 1.0 |
| E3 | Image resolution limit (1,310,720 pixels) | 41.4 | 3.9 / 8.2 | 4.2 | 70 | 2.1 | 4 | 2 | 0.5 |
| E4 | Parallel tool calls | 41.6 | 3.9 / 8.2 | 4.2 | 71 | 2.1 | 4 | 2 | 0.5 |
| E5 | FP8 weights and FP8 KV cache | 43.3 | 8.2 / 17.3 | 4.2 | 72 | 2.0 | 8 | 2 | 0.25 |
| E6 | N-gram speculative decoding (two lowest rates) | 44.5 | 1.4 / 2.8 | 4.2 | 73 | 2.0 | 1 | 2 | 2.0 |
| E7 | 4-bit page retriever on the generator's GPU, compiled | 42.7 | 3.0 / 6.0 | 4.2 | 71 | 2.1 | 4 | 1 | 0.25 |

E6 was run at the two lowest rates only, so its goodput and GPU-seconds per task are not comparable with the other rows.

### Scale-out and routing (E8, E7 configuration, one repeat)

| Replicas | Rate (tasks/s) | Policy | Prefix hit rate (%) | p95 task time (s) | Tasks within objective |
|---|---|---|---|---|---|
| 2 | 4 | round-robin | 60 | 4.2 | 100% |
| 2 | 4 | affinity | 71 | 3.9 | 100% |
| 2 | 4 | prefix | 74 | 3.8 | 100% |
| 2 | 8 | round-robin | 60 | 7.4 | 100% |
| 2 | 8 | affinity | 70 | 5.7 | 100% |
| 2 | 8 | prefix | 75 | 5.6 | 100% |
| 2 | 16 | round-robin | 48 | 415.5 | 8% |
| 2 | 16 | affinity | 46 | 325.1 | 1% |
| 2 | 16 | prefix | 66 | 74.0 | 13% |
| 4 | 8 | round-robin | 53 | 4.8 | 100% |
| 4 | 8 | affinity | 71 | 4.5 | 100% |
| 4 | 8 | prefix | 83 | 4.2 | 100% |
| 4 | 16 | round-robin | 53 | 34.8 | 68% |
| 4 | 16 | affinity | 71 | 30.0 | 69% |
| 4 | 16 | prefix | 82 | 33.1 | 70% |
| 4 | 32 | round-robin | 51 | 325.7 | 2% |
| 4 | 32 | affinity | 60 | 310.4 | 2% |
| 4 | 32 | prefix | 79 | 307.0 | 1% |

### Choosing the GPU (E9, E7 configuration)

| GPU | Goodput (tasks/s) | p50 / p95 at goodput (s) | GPU-seconds per task |
|---|---|---|---|
| NVIDIA L40S (48 GB) | 1 | 7.5 / 15.3 | 1.0 |
| NVIDIA A100-SXM4-80GB | 1 | 5.4 / 13.5 | 1.0 |
| NVIDIA H100 80GB HBM3 (SXM) | 4 | 3.0 / 6.0 | 0.25 |

The E7 setting `--gpu-memory-utilization=0.80` does not fit on one L40S next to the 4-bit retriever (36.2 of 47.4 GiB
free at vLLM start-up, 37.9 GiB requested), so the L40S row uses 0.70. L40S and A100: one repeat per rate (0.5, 1 and
2 tasks/s; at 2 tasks/s 8% and 53% of tasks met the objective); H100: three repeats (E7). Files: `results/e9_modal/`.

### Regression gate and batch invariance

| Test | Result |
|---|---|
| R1: run-to-run variation | coefficient of variation of p50 task time over 3 repeats: 1.5% |
| R1: vLLM 0.32.0 to 0.33.0, E7 at 4 tasks/s | PASS (p50 3.00 vs 2.99 s, p95 6.00 vs 5.79 s, accuracy 42.7 vs 41.6%) |
| R1: Chapter 8 benchmark (Llama-3.1-8B), 0.32.0 to 0.33.0 | goodput within 0.2% at every rate from 1 to 48 req/s |
| R2: `VLLM_BATCH_INVARIANT=1` on E7 | goodput change 0% (4 tasks/s in both); p50 at goodput +14%; identical answers across load 18.8% (17.9% without) |

Answers repeat between runs at 0.5 tasks/s in 53-59% of tasks for E2-E4 and in 20-26% from E5 on (FP8 weights and FP8
KV cache), with or without batch invariance.

### Result files

| Path | Contents |
|---|---|
| `results/case_study.json` | all values above, in the schema of `report.py` |
| `results/E*_H100_single_<rate>_<repeat>.jsonl` | one line per task: arrival, task time, steps, tokens, cached tokens, images, answer |
| `results/*.judged.json` | per-task accuracy verdicts at goodput |
| `results/E8_*`, `results/R1_*`, `results/R2.json`, `results/R0*.json` | scale-out, regression gate, batch invariance, retrieval |
| `results/e9_modal/`, `results/E9.json` | GPU choice (L40S and A100 on Modal) |
| `results/judge_agreement.json` | second-judge and human check of the verdicts |

Traces (`*_trace.jsonl`) are not in the repository.

## Online resources

The print chapter refers to the material below. Reference numbers are those of the chapter.

- Full scale-out and GPU-choice results: [Scale-out and routing (E8)](#scale-out-and-routing-e8-e7-configuration-one-repeat)
  and [Choosing the GPU (E9)](#choosing-the-gpu-e9-e7-configuration) above.
- Running the routing comparison on Kubernetes with llm-d 1.0 and four vLLM pods: [`k8s/`](k8s/README.md). llm-d
  replaces the case study's router with its precise prefix-cache scorer and adds flow control and autoscaling [6].

### Cluster serving frameworks (November 2026)

Features change with each release, and all three projects release about monthly.

| | llm-d 1.0 | NVIDIA Dynamo 1.6 | KServe 0.18 |
|---|---|---|---|
| Governance, license | CNCF Sandbox, Apache 2.0 | NVIDIA-led, Apache 2.0 | CNCF Incubating, Apache 2.0 |
| Interface | Gateway API Inference Extension | Own frontend and CRDs; can plug into the Gateway API | `LLMInferenceService` CRD |
| Routing | Filter, score, pick; precise prefix-cache scoring from KV events; latency predictor | KV-aware router; flow control | Through llm-d |
| Prefill/decode pools | Yes | Yes, with planner | Through llm-d |
| KV offload | CPU, SSD; peer-to-peer prefix sharing | GPU, CPU, SSD, object storage | Through llm-d |
| Engines | vLLM, SGLang, TensorRT-LLM | vLLM, SGLang, TensorRT-LLM | vLLM and others |

### Failure modes of a multi-replica LLM service

| Failure | Effect without a mechanism | Mechanism | Cost |
|---|---|---|---|
| Replica crash | In-flight requests fail | Health checks; request migration; client retry | Recomputed prefill for migrated requests |
| Overload | Queues grow and most requests miss their objectives | Admission control and load shedding (503 with Retry-After); priority bands | Rejected requests |
| Router failure | Service unavailable | Redundant routers; hot standby | Extra router instances |
| Cold start | Minutes before a new replica serves | Weights cached on local disk or CPU memory; sleep mode; weight streaming | Memory or storage |
| Spot preemption | Replica lost with short notice | Migration to on-demand replicas; checkpointed state | Lower savings |
| Upgrade | Capacity drops during rollout; regressions | Rolling or canary upgrades; regression gates | Duplicate capacity during rollout |

### Architecture choices and their reported effect on inference cost

As measured by the authors of each paper.

| Design | Example | Reported effect |
|---|---|---|
| Multi-head latent attention + MoE | DeepSeek-V2 (236B total, 21B active) | 93.3% smaller KV cache and 5.76× higher maximum generation throughput than DeepSeek 67B [46] |
| Local:global attention 5:1, 1,024-token window | Gemma 3 | KV memory overhead at 32K context reduced from about 60% to under 15% [43] |
| Attention:Mamba 1:7 with MoE | Jamba | 4 GB KV cache at 256K context against 32 GB for Mixtral; 3× Mixtral's throughput at 128K [44] |
| Hybrid Mamba-Transformer | Nemotron-H 8B and 56B | Up to 3× faster inference than similarly sized Transformers at similar accuracy [47] |
| Blockwise NAS on a trained model | Nemotron-51B from Llama-3.1-70B | Up to 2.17× throughput on one H100 with about 98.4% of the parent's accuracy [48] |

### GPU-seconds and cost per task for the case study

The chapter reports GPU-seconds per task: GPUs / goodput, at full utilization. The cost per task at your own price is
price per GPU-hour x GPUs / (goodput x utilization x 3,600), or GPU-seconds per task x price per GPU-hour / 3,600 at
full utilization (`cost.py`).

| Experiment | GPUs | Goodput (tasks/s) | GPU-seconds per task |
|---|---|---|---|
| E0, E1 | 2 x H100 | 1 | 2.0 |
| E2 | 2 x H100 | 2 | 1.0 |
| E3, E4 | 2 x H100 | 4 | 0.5 |
| E5 | 2 x H100 | 8 | 0.25 |
| E6 | 2 x H100 | at least 1 (tested to 1) | at most 2.0 |
| E7 | 1 x H100 | 4 | 0.25 |
| E9 L40S (Modal) | 1 | 1 | 1.0 |
| E9 A100 80GB (Modal) | 1 | 1 | 1.0 |
| E9 H100 (E7) | 1 | 4 | 0.25 |

[`cost.py`](cost.py) prints 2.0 and 5.0 GPU-seconds per task for one GPU at 0.5 tasks/s, with utilization 1.0 and 0.4
(a price of 1 per GPU-hour, times 3,600).

### H100 price trend

The median on-demand H100 price across 46 providers rose by about 32% between November 2025 and November 2026
(GetDeploying, "GPU Price Trends", weekly median on-demand prices across providers, accessed 7 November 2026,
https://getdeploying.com/gpu-price-trends).

### Status of the OpenTelemetry GenAI semantic conventions

As of November 2026 the OpenTelemetry semantic conventions for generative AI had Development status, so attribute
names could still change between versions (OpenTelemetry, "Semantic Conventions for Generative AI Systems", accessed
7 November 2026, https://opentelemetry.io/docs/specs/semconv/gen-ai/).

### Serving-framework landscape notes (November 2026)

- llm-d 1.0 was a CNCF Sandbox project (llm-d Project, "llm-d v1.0: State-of-the-Art Inference, Ready for Day 2",
  October 2026, https://llm-d.ai/blog/llm-d-v1.0-state-of-the-art-inference-ready-for-day-2).
- Hugging Face put its TGI server in maintenance mode and recommends vLLM and SGLang (Hugging Face, "Text Generation
  Inference" documentation, accessed 25 October 2026, https://huggingface.co/docs/text-generation-inference/index).
- Amazon reported that its Rufus assistant runs on vLLM across tens of thousands of Trainium chips (Amazon Web
  Services, "How Amazon Scaled Rufus by Building Multi-Node Inference Using AWS Trainium Chips and vLLM", AWS Machine
  Learning Blog, August 2025,
  https://aws.amazon.com/blogs/machine-learning/how-amazon-scaled-rufus-by-building-multi-node-inference-using-aws-trainium-chips-and-vllm).
- The feature comparison of llm-d, Dynamo and KServe above is dated November 2026.

### Disaggregated prefill status in vLLM

vLLM's documentation (vLLM 0.33.0, November 2026) marked disaggregated prefill as experimental and subject to change
(vLLM Project, "Disaggregated Prefilling (experimental)", vLLM documentation v0.33.0, accessed 25 November 2026,
https://docs.vllm.ai/en/v0.33.0/features/disagg_prefill/).

### Batch invariance in vLLM 0.33.0

vLLM provided batch invariance as a beta feature (`VLLM_BATCH_INVARIANT=1`) on NVIDIA GPUs of compute capability 8.0
and newer and on recent AMD GPUs. With tensor parallelism it disabled some custom all-reduce implementations (vLLM
Project, "Batch Invariance", vLLM documentation, accessed 7 November 2026,
https://docs.vllm.ai/en/latest/features/batch_invariance/).
