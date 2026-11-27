# Chapter 10: Efficient Model Adaptation and Maintenance

**Status: case study run on 2026-11-20 to 2026-11-23.** H100 results are in `results/ch10_results.json`, the file the
chapter build reads; T4 and A100 results (Modal) are in `results/modal/` and listed below.

Companion code for Chapter 10. The case study adapts the RAG pipeline's generators (Gemma-2B-IT and
Llama-3.1-8B-Instruct) to the HotpotQA multi-hop task and measures what each adaptation method costs and changes:
training memory and time, answer quality, abstention, forgetting, KL divergence from the base model, and serving
speed with merged and unmerged LoRA adapters in vLLM 0.33.0. It also distills the adapted Llama-3.1-8B into
Llama-3.2-1B and re-bases the same LoRA recipe onto the next generator, Qwen3-VL-8B-Instruct.

## Scripts

| Script | What it does |
|---|---|
| `common.py` | Shared settings: models, GPUs, paths, package versions |
| `modal_app.py` | Modal application; every GPU step is an entrypoint (`smoke`, `prep`, `train`, `merge`, `evaluate`, `serve_bench`, `distill`, `spectrum`, `collect`) |
| `smoke_adapters.py` | Untrained rank-16 LoRA adapters (B = 0) for the vLLM smoke test |
| `prep_data.py` | Builds the training and evaluation datasets |
| `train_peft.py` | Fine-tunes with LoRA, QLoRA, DoRA, PiSSA or full fine-tuning |
| `train_memory.py` | Static training memory per method for both generators |
| `train_time.py` | Lower-bound fine-tuning time from FLOPs (Code listing in the chapter) |
| `merge_adapter.py` | Merges a LoRA/QLoRA adapter into the 16-bit base model |
| `eval_quality.py` | Quality, abstention, forgetting and KL evaluation with vLLM |
| `serve_bench.py` | Serving benchmark for merged and unmerged LoRA adapters with vLLM 0.33 |
| `distill.py` | Distillation arms (SFT, sequence-level KD, token-level KD, GKD) |
| `svd_spectrum.py` | Singular values of the weight change of one MLP projection of Gemma-2B (online spectrum figure, below) |
| `collect_results.py` | Collects all measurements into `results/ch10_results.json`, the file the chapter build reads |

## Experimental setup

| Item | Value |
|---|---|
| Hardware | NVIDIA H100 80GB HBM3 (SXM, 700 W), driver 590.48.01, on a GPU cluster with 8 H100 GPUs per node (NVLink 4 through NVSwitch). One GPU per run, two for Llama-3.1-8B full fine-tuning (FSDP) |
| Serving and evaluation | vLLM 0.33.0, torch 2.14.0+cu130, transformers 5.20.2, lm-evaluation-harness 0.4.13 |
| Training | torch 2.14.0, transformers 5.23.0, peft 0.21.2, trl 1.17.1, bitsandbytes 0.50.2, accelerate 1.16.0 |
| Models | google/gemma-2b-it, meta-llama/Llama-3.1-8B-Instruct, meta-llama/Llama-3.2-1B-Instruct (student), Qwen/Qwen3-VL-8B-Instruct (re-base) |
| Training data | 10,000 HotpotQA (distractor) RAG prompts; 20% have their supporting passages removed and the target "The context does not contain the answer." (RAFT style) |
| Training recipe | 2 epochs, sequence length 1,024 with packing, batch 8 x 2 accumulation, cosine schedule, 3% warm-up, seed 0. Learning rate 2e-4 (Gemma-2B LoRA/QLoRA), 1e-4 (Llama-3.1-8B and Qwen3-VL-8B LoRA/QLoRA), 2e-5 (full fine-tuning). LoRA r = 16, alpha = 2r, all linear layers (language model only for Qwen3-VL) |
| HotpotQA | exact match on the first 1,000 items of the evaluation split, greedy decoding |
| Abstention | F1 of "not in context" on 500 SQuAD 2.0 items (answerable and unanswerable) |
| Forgetting | lm-evaluation-harness through vLLM: GSM8K chain of thought 8-shot (250 items), IFEval (541, prompt-level strict), ARC-Challenge 0-shot (500), MMLU 5-shot (14,042) |
| KL | per-token KL from the base model on the base model's top-20 tokens, 200 RAG prompts (rows 800-999 of the Chapter 8 workload) |
| Serving | `vllm serve`, 500 HotpotQA prompts, Poisson arrivals at 4-192 req/s for 60 s each; objective TTFT <= 0.5 s and TPOT <= 50 ms; capacity = highest rate with >= 90% of requests within the objective |
| Knowledge probe | 200 QA pairs about 60 Wikipedia articles created after 2024-01-01, closed book and with the article; LoRA trained 3 epochs on the article text |

Differences and paired standard errors (± values) are computed per item against the base model on the same items.

Notes:

- Gemma-2B-IT's chat template has no system role. For Gemma the system instruction is placed at the start of the
  first user turn (`common.for_model`), and generation stops at `<end_of_turn>` (`common.generation_config_dir`),
  because gemma-2b-it's generation config lists only `<eos>` as a stop token and a fine-tuned model otherwise starts
  a new turn after its answer. Llama and Qwen use their templates unchanged.
- `train_peft.py` appends each example's `answer` as the assistant turn before training.
- Evaluation sizes are smaller than the full splits for HotpotQA, abstention, GSM8K and ARC-C (see the table above).
- A QLoRA adapter is evaluated unmerged on the BF16 base; merged into BF16 it gives a KL of 0.696 against 0.695
  unmerged and HotpotQA EM 62.2 against 62.1 (Llama-3.1-8B).
- The student TPOT in the distillation table is measured on the H100. TPOT depends on the architecture and size of
  the model, so the A100 values can be measured on the base model of each student.

## How to run

On Modal (every GPU step is an entrypoint of `modal_app.py`; the Hugging Face token is read only from the Modal
secret `huggingface`):

```bash
modal run chapter10/modal_app.py::smoke        # vLLM 0.33 smoke test on H100, A100-80GB and T4
modal run chapter10/modal_app.py::prep         # data
modal run chapter10/modal_app.py::train --model gemma-2b --method lora --extra "--lr 2e-4"
modal run chapter10/modal_app.py::evaluate --model gemma-2b --run gemma-2b-lora-r16-s0-H100 --task hotpot --limit 1000
# merge, serve_bench, distill, spectrum, collect: see RUN_PROMPT_ch10.md
```

The run on 2026-11-20 used the same scripts on a GPU cluster (batch jobs, one job per run), with the Modal volume replaced by a shared directory (`CH10_VOL`). `train_peft.py` (`--max-examples`, `--max-steps`), `eval_quality.py` (`--limit`, `--forget-tasks`) and
`distill.py` (`--max-examples`) take size flags for short test runs.

## Results

Base models (H100, vLLM 0.33.0): HotpotQA EM 35.7 (Llama-3.1-8B) and 0.1 (Gemma-2B, which answers in sentences
without fine-tuning); abstention F1 74.9 and 40.3; GSM8K 80.8 and 5.6; IFEval 74.1 and 23.7; ARC-C 51.2 and 35.2;
MMLU 68.7 and 34.3.

### Training cost (H100)

| Model | Method | Peak memory (GB) | Tokens/s | GPU-hours |
|---|---|---|---|---|
| Gemma-2B | LoRA r=16 | 8.4 | 10,694 | 0.35 |
| Gemma-2B | QLoRA r=16 | 4.7 | 11,113 | 0.33 |
| Gemma-2B | Full FT | 24.4 | 12,565 | 0.29 |
| Llama-3.1-8B | LoRA r=16 | 21.5 | 4,855 | 0.77 |
| Llama-3.1-8B | QLoRA r=16 | 10.0 | 5,070 | 0.74 |
| Llama-3.1-8B | Full FT (FSDP, 2 GPUs) | 71.2 | 7,846 | 0.95 |

Cost of a run = GPU-hours x your price per GPU-hour.

### Quality, paired differences against the base model (points)

| Model | Variant | HotpotQA EM | Abstention F1 | GSM8K | IFEval | ARC-C | MMLU | Forgetting (mean) | KL vs base |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | LoRA r=16 | +26.4 ± 1.7 | -8.4 | -8.4 ± 1.9 | -13.1 ± 2.1 | -2.4 ± 1.9 | -1.7 ± 0.3 | -6.4 ± 0.8 | 0.877 |
| Llama-3.1-8B | LoRA r=16 + 10% general data | +26.0 ± 1.7 | -8.9 | -14.2 ± 2.1 | -21.6 ± 2.2 | -2.4 ± 1.7 | -2.0 ± 0.3 | -10.1 ± 0.9 | 0.819 |
| Llama-3.1-8B | QLoRA r=16 | +26.4 ± 1.7 | -12.0 | -6.4 ± 1.7 | -8.9 ± 1.9 | -1.8 ± 1.8 | -1.7 ± 0.3 | -4.7 ± 0.8 | 0.695 |
| Gemma-2B | LoRA r=16 | +52.6 ± 1.6 | +11.8 | -0.2 ± 1.5 | -5.2 ± 2.0 | -2.0 ± 1.9 | +1.0 ± 0.4 | -1.6 ± 0.8 | 0.920 |
| Gemma-2B | QLoRA r=16 | +50.0 ± 1.6 | +20.4 | +1.4 ± 1.6 | -4.6 ± 1.9 | -1.6 ± 1.9 | -0.6 ± 0.4 | -1.4 ± 0.8 | 0.871 |
| Gemma-2B | Full FT | +55.1 ± 1.6 | +22.1 | +0.6 ± 1.4 | -4.4 ± 1.8 | +1.0 ± 1.8 | +1.8 ± 0.4 | -0.3 ± 0.7 | 0.938 |

Forgetting (mean) averages GSM8K, IFEval, ARC-C and MMLU (5-shot, 14,042 items). In the rank sweep below it averages GSM8K, IFEval and ARC-C.

### Rank sweep (Gemma-2B LoRA, differences against the base model)

| Rank | HotpotQA, α/r | Forgetting, α/r | HotpotQA, rsLoRA | Forgetting, rsLoRA |
|---|---|---|---|---|
| 4 | +50.3 ± 1.6 | -3.8 ± 1.0 | +51.8 ± 1.6 | -3.3 ± 1.0 |
| 16 | +52.6 ± 1.6 | -2.5 ± 1.0 | +52.0 ± 1.6 | -4.4 ± 1.0 |
| 64 | +54.3 ± 1.6 | -3.6 ± 1.1 | +45.6 ± 1.6 | -9.9 ± 1.1 |
| 256 | +54.9 ± 1.6 | -5.6 ± 1.0 | -0.1 ± 0.1 | -10.5 ± 1.2 |

rsLoRA at rank 256 diverged at the shared learning rate of 2e-4 (its scale factor alpha/sqrt(r) grows with the rank).

### Knowledge probe

| Llama-3.1-8B-Instruct | Accuracy (%) |
|---|---|
| Base, closed book | 3.0 |
| Base with retrieved articles | 79.0 |
| LoRA on the articles, closed book | 2.5 |
| LoRA on the articles, with retrieved articles | 79.5 |

### Serving (Llama-3.1-8B, vLLM 0.33.0, H100)

| Configuration | TPOT, 1 request (ms) | Capacity (req/s) | TTFT p90 at capacity (ms) |
|---|---|---|---|
| Base model | 6.47 | 64 | 53 |
| Merged adapter | 6.45 | 64 | 53 |
| Unmerged, 1 adapter | 7.86 | 48 | 71 |
| Unmerged, 4 adapters | 8.18 | 32 | 145 |
| Unmerged, 16 adapters | 8.23 | 24 | 186 |

### Distillation (teacher: merged Llama-3.1-8B LoRA)

| Model | HotpotQA EM | IFEval (prompt, strict) | GPU-hours | TPOT on H100 (ms) |
|---|---|---|---|---|
| Llama-3.1-8B + LoRA (teacher) | 62.1 | 61.0 | 0.77 | 6.45 |
| Llama-3.2-1B, SFT on reference answers | 49.2 | 42.7 | 0.15 | 1.48 |
| Llama-3.2-1B, sequence-level KD | 51.1 | 43.8 | 0.15 | 1.48 |
| Llama-3.2-1B, token-level KD | 49.5 | 39.6 | 0.35 | 1.48 |
| Llama-3.2-1B, on-policy KD | 51.1 | 46.0 | 0.30 | 1.48 |
| Gemma-2B, sequence-level KD | 52.5 | 11.8 | 0.24 | 2.46 |
| Gemma-2B, SFT on reference answers | 53.1 | 10.9 | 0.23 | 2.45 |

### Re-base to the next generator

| Generator | HotpotQA EM, base | HotpotQA EM, LoRA | GPU-hours to adapt | TPOT base / with unmerged adapter (ms) |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 35.7 | 62.1 | 0.77 | 6.47 / 7.86 |
| Qwen3-VL-8B-Instruct | 59.6 | 63.1 | 0.89 | 6.83 / 8.58 |

Data for the online spectrum figure (singular values of the weight change of layer 9's MLP down projection of Gemma-2B for full
fine-tuning, LoRA r = 16 and LoRA r = 256) is in `results/spectrum.json`.

### Additional runs

Seeds (H100, same recipe):

| Model | Method | HotpotQA EM, seeds 0 / 1 / 2 | Mean ± SD |
|---|---|---|---|
| gemma-2b | Lora | 52.7 / 52.4 / 52.6 | 52.6 ± 0.2 |
| gemma-2b | QLORA | 50.1 / 51.8 / 50.9 | 50.9 ± 0.9 |
| llama-3.1-8b | Lora | 62.1 / 62.0 / 62.7 | 62.3 ± 0.4 |
| llama-3.1-8b | QLORA | 62.1 / 62.3 / 61.5 | 62.0 ± 0.4 |

Learning rate (Gemma-2B LoRA r = 16, 2 epochs):

| Learning rate | HotpotQA EM | Final training loss |
|---|---|---|
| 5e-5 | 47.5 | 1.81 |
| 1e-4 | 50.1 | 1.76 |
| 2e-4 | 52.9 | 1.71 |

LoRA variants (differences against the base model; DoRA is evaluated after merging, because vLLM does not serve DoRA
adapters; PiSSA is converted to a rank-32 LoRA on the original weights with PEFT's PiSSA conversion):

| Method (Gemma-2B, r = 16) | HotpotQA EM | Abstention F1 | GSM8K | IFEval | ARC-C | KL | Peak memory (GB) | Tokens/s |
|---|---|---|---|---|---|---|---|---|
| LoRA r=16 | +52.6 ± 1.6 | +11.8 | -0.2 ± 1.5 | -5.2 ± 2.0 | -2.0 ± 1.9 | 0.920 | 8.4 | 10,694 |
| DoRA r=16 (merged) | +52.6 ± 1.6 | +13.0 | -1.0 ± 1.4 | -2.8 ± 2.0 | -2.6 ± 1.9 | 0.928 | 11.0 | 6,085 |
| PiSSA r=16 (converted to LoRA r=32) | +53.0 ± 1.6 | +24.2 | -3.6 ± 1.4 | -7.4 ± 2.0 | -4.4 ± 2.0 | 1.011 | 8.4 | 10,719 |

Llama-3.1-8B full fine-tuning (2 H100, FSDP): HotpotQA EM 65.6 (+29.9 ± 1.8), abstention F1 -2.6,
GSM8K -13.8 ± 2.0, IFEval -44.7 ± 2.5, ARC-C -3.0 ± 2.0, KL 1.12.

### T4 and A100 runs (Modal)

| Run | GPU | Result |
|---|---|---|
| Gemma-2B QLoRA, 100 steps | Tesla T4 (FP16) | 4.73 GB peak, 849 tokens/s |
| Llama-3.1-8B LoRA, 200 steps | NVIDIA A100-SXM4-80GB | 21.4 GB peak, 2,400 tokens/s |
| Llama-3.1-8B QLoRA, 200 steps | NVIDIA A100-SXM4-80GB | 9.9 GB peak, 2,443 tokens/s |
| Student TPOT, Llama-3.2-1B architecture | NVIDIA A100-SXM4-80GB | 2.56 ms |
| Student TPOT, Gemma-2B architecture | NVIDIA A100-SXM4-80GB | 4.40 ms |
| Gemma-2B serving, base and merged adapter | Tesla T4 (FP16) | TPOT 21.2 ms; capacity 3 req/s; TTFT p90 about 300 ms |

Files: `results/modal/` (training metrics, serving benchmarks, smoke tests, GPU records). The student TPOT is measured on
the base model of each student's architecture; TPOT depends on the architecture and size, not on the weight values.
On the T4, `train_peft.py` trains in FP16 (BF16 matmuls need compute capability 8.0) with FP32 trainable weights.

### Result files

| Path | Contents |
|---|---|
| `results/ch10_results.json` | all tables above, in the schema of `collect_results.py` |
| `results/ch10_phase2.json` | seeds, learning rates, DoRA, PiSSA, Llama full fine-tuning |
| `results/eval/` | per-item HotpotQA, abstention and probe results for each run |
| `results/lmeval/<run>/` | lm-evaluation-harness results and per-sample logs (GSM8K, IFEval, ARC-C) |
| `results/serve/` | serving benchmarks, including the TPOT of each student |
| `results/spectrum.json`, `results/smoke_H100.json` | singular values for the online spectrum figure; vLLM 0.33 smoke test |

Every result file carries an `env` block with the GPU name from `nvidia-smi`, the driver and the package
versions. The per-token KL files (about 70 MB each) and the MMLU per-sample logs (1.4 GB) are not in the repository; their
values are in `ch10_results.json`.

## Online resources

The print chapter refers to the material below.

### Rank sweep on Gemma-2B

Change in HotpotQA exact match and in the forgetting-suite average, in points, with the original α / r scaling and
with rsLoRA (α = 2r), learning rate 2e-4. Data: `results/ch10_results.json` (key `rank`).

| Rank | α / r: HotpotQA | α / r: forgetting | rsLoRA: HotpotQA | rsLoRA: forgetting | Adapter size (MB) |
|---|---|---|---|---|---|
| 4 | +50.3 ± 1.6 | −3.8 ± 1.0 | +51.8 ± 1.6 | −3.3 ± 1.0 | 10 |
| 16 | +52.6 ± 1.6 | −2.5 ± 1.0 | +52.0 ± 1.6 | −4.4 ± 1.0 | 39 |
| 64 | +54.3 ± 1.6 | −3.6 ± 1.1 | +45.6 ± 1.6 | −9.9 ± 1.1 | 157 |
| 256 | +54.9 ± 1.6 | −5.6 ± 1.0 | −0.1 ± 0.1 | −10.5 ± 1.2 | 627 |

With α / r, exact match rises with the rank and forgetting grows from rank 16 upward. rsLoRA multiplies the update by
α / √r, which at rank 256 is 16 times the α / r value; at the same learning rate the run at rank 256 did not train.
rsLoRA at high rank needs its own learning-rate sweep.

### Singular values of the weight update

![Singular values of the weight change of one MLP projection of Gemma-2B](online/lora_update_spectrum.png)

Singular values of the weight change ΔW of one MLP projection of Gemma-2B after full fine-tuning and after LoRA at
ranks 16 and 256, normalized to the largest. The LoRA updates have exactly r non-zero singular values: the 17th
singular value of the rank-16 update is 0.4% of the largest. The full fine-tuning update decays slowly, and its 512th
singular value is 34% of the largest. Script: `svd_spectrum.py`; data: `results/spectrum.json`.

### Variants of LoRA

Reported results are from the original papers, under their conditions. Reference numbers are those of the chapter.

| Method | What changes | Training memory | Reported result | How to enable |
|---|---|---|---|---|
| rsLoRA [16] | Scaling α/√r | Same as LoRA | Stable training at high rank | `use_rslora=True` |
| DoRA [21] | Separate magnitude vector; LoRA updates the direction | Slightly higher | Higher accuracy than LoRA at equal rank on commonsense tasks; merge for inference | `use_dora=True` |
| LoRA+ [22] | Larger learning rate for B than for A | Same as LoRA | Up to 2× faster convergence | `create_loraplus_optimizer` |
| PiSSA [23] | Initializes A and B from the top singular vectors of W₀ | Same as LoRA | Mistral-7B GSM8K 72.9 vs 67.7 with LoRA | `init_lora_weights="pissa"` |
| VeRA [24] | Shared frozen random A and B; trains scaling vectors | Far fewer trainable parameters | Comparable accuracy with 10× or more fewer parameters | `VeraConfig` |
| LoRA-FA [25] | Freezes A | Lower activation memory | LLaMA-7B memory 56 → 27.5 GB | `create_lorafa_optimizer` |
| GaLore [26] | Full-rank weights; low-rank projection of gradients | Optimizer states up to 65.5% smaller | 7B pre-training on a 24 GB GPU | Trainer `optim="galore_adamw"` |

### Structured pruning of transformer language models

| Method | Unit removed | Recovery training | Result |
|---|---|---|---|
| Michel et al. [45] | Attention heads | None | 40% of BERT's heads with no noticeable loss on MNLI |
| ShortGPT [46] | Layers, by input-output similarity | None | 10 of 40 layers of Llama-2-13B, about 92% of benchmark performance kept |
| Gromov et al. [47] | Contiguous deep layers | Short QLoRA fine-tune, one GPU | Up to half of the layers of large Llama-2 models with little MMLU loss |
| Sheared-LLaMA [48] | Layers, heads, width | 50B tokens | Llama-2-7B to 1.3B and 2.7B, about 3% of the compute of training from scratch |
| Minitron [49], [51] | Width or depth | 94B tokens, distillation | Llama-3.1-8B to 4B: MMLU 60.5 (width) or 58.7 (depth) vs 65.3; about 1.8× and 2.7× faster in TensorRT-LLM |
