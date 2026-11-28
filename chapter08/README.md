# Chapter 8 — High-Performance Inference and Serving

Companion code for Chapter 8. The analytical scripts run on any machine; the case study serves the RAG pipeline's generator (Gemma-2B-IT, FP16) with vLLM and measures it under load on an A100 80GB, an H100 and a T4, and serves Llama-3.1-8B-Instruct (BF16) on the A100 and H100 to show the regime where the KV cache limits the batch.

## What you compute and measure

1. KV cache bytes per token and token capacity for any Hugging Face model (`kv_calc.py`).
2. The bookkeeping of a paged KV cache: block tables, reference counts, copy-on-write and prefix reuse (`paged_kv.py`).
3. Attention computed tile by tile with an online softmax, and the split-KV combine of Flash-Decoding (`tiled_attention.py`).
4. Expected tokens per target pass and the modeled speedup of speculative decoding (`spec_math.py`).
5. A Roofline estimate of the prefill and decode GPUs a workload needs, for disaggregated serving (`pd_size.py`).
6. The case study: single-request latency, a load sweep with goodput, a long-context study (4K prompts, 1K answers) with KV usage and preemptions, prefix caching on and off, the prefill token budget, and n-gram speculative decoding (`modal_serving.py`).

## Run it

```bash
python chapter08/paged_kv.py
python chapter08/spec_math.py
python chapter08/pd_size.py
python chapter08/tiled_attention.py                     # needs PyTorch (CPU is fine)
python chapter08/kv_calc.py google/gemma-2b-it --weight-gb 5.0

# case study on Modal (needs a Modal secret named "huggingface" whose token can read
# google/gemma-2b-it and meta-llama/Llama-3.1-8B-Instruct)
modal run chapter08/modal_serving.py                    # all GPUs and models, all studies
modal run chapter08/modal_serving.py --gpus T4 --models gemma --studies single,sweep
python chapter08/summarize_results.py                   # tables and results/sweep.png
```

To run the case study on your own GPU instead, build the workload once and point the load generator at a running server:

```bash
python chapter08/build_prompts.py --out-dir data --num 1000
# The load studies repeat prompts, so prefix caching is off, as in the book's runs
vllm serve google/gemma-2b-it --dtype float16 --max-model-len 2048 --no-enable-prefix-caching
python chapter08/loadgen.py --model google/gemma-2b-it --prompts data/prompts_single.jsonl \
    --rate 4 --num 200 --max-tokens 128 --slo-ttft-ms 500 --slo-tpot-ms 50
```

## Files

| File | What it does |
|---|---|
| `kv_calc.py` | KV bytes per token from a model config; tokens that fit in the memory left after the weights |
| `paged_kv.py` | Teaching model of a paged KV cache manager (no tensors, runs anywhere) |
| `tiled_attention.py` | Online-softmax attention and split-KV decoding, checked against the reference |
| `pd_size.py` | Prefill and decode pool sizes for a workload from the Roofline step time and KV capacity |
| `spec_math.py` | Speculative decoding: expected tokens per pass and speedup against acceptance rate and draft length |
| `build_prompts.py` | Builds the RAG serving workload (1,000 distinct SQuAD questions, BGE-M3 + FAISS top-5 over the Chapter 2 corpus), a follow-up turn for each, and 4,096-token long-context prompts |
| `loadgen.py` | Open-loop Poisson load generator for OpenAI-compatible servers; TTFT, ITL, TPOT, E2E percentiles and goodput |
| `modal_serving.py` | Runs the case-study studies on Modal, one fresh `vllm serve` per configuration; writes `results/<gpu>_<model>.json` |
| `summarize_results.py` | Prints the case-study tables and plots the load sweep |

## What to expect when you reproduce the results

Your numbers will differ from the book's, and some rows will differ more than others. Measured on the same models in
October and November 2026:

| Quantity | How much it moved between setups | Why |
|---|---|---|
| Eager and `torch.compile` time per output token at batch size 1 (Table 8.5) | Up to about 2x. Gemma-2B eager on an A100 80GB: 16.5 ms (FP16, 20 output tokens, Modal) and 25.2 ms (BF16, 128 output tokens, Colab Pro, Chapter 11). On an H100: 11.3 and 18.7 ms in two Modal sessions with the same script | Limited by CPU-side overhead: host CPU, PyTorch and transformers versions, data type and timing harness all change it |
| vLLM time per output token at low load | Within 2% (Gemma-2B) and 5% (Llama-3.1-8B) between Chapters 8 and 11 | Close to the weight-read floor, so set mainly by memory bandwidth |
| Capacity and goodput under load | Depends on the GPU's power limit and clock under sustained load | The A100 ran at 1,245 to 1,320 MHz instead of 1,410 MHz during the load sweeps |

Compare the ordering of the stacks and each value's ratio to the weight-read floor. Record the GPU, driver, host CPU
and package versions with every result (the scripts in this folder do), and repeat each measurement at least three
times before comparing it with the book.

## Versions

The case study pins vLLM 0.32.0 in its own image (vLLM pins its own PyTorch build). Set `CH08_VLLM` to try another version, and record the version with your results: serving engines change their defaults between releases.

## Online resources

The print chapter refers to the material below.

- Knowledge check answers: [`knowledge_check_solutions.md`](knowledge_check_solutions.md).
- The list of the book's 51 Performance Engineering Principles, with the chapter that states each one:
  [`docs/principles.md`](../docs/principles.md).

### Open-source LLM serving engines (October 2026)

Versions and features change from release to release; check each project's documentation.

| Engine | Status | Hardware | Techniques from Chapter 8 |
|---|---|---|---|
| vLLM | PyTorch Foundation hosted project since May 2025 ([PyTorch Foundation, 7 May 2025](https://pytorch.org/blog/pytorch-foundation-welcomes-vllm/)); V1 engine only since release 0.11 ([v0.11.0 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.11.0)); release 0.32 in October 2026 ([v0.32.0 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)) | NVIDIA, AMD, Intel and CPU; TPU and Ascend through plugins | Continuous batching, PagedAttention, hash-based prefix caching, chunked prefill, speculative decoding (n-gram, draft model, EAGLE, MTP, suffix), structured output, expert parallelism, P/D disaggregation through KV connectors, multi-LoRA |
| SGLang | Incubated at LMSYS; RadixArk, founded by SGLang's creators and core maintainers, is the commercial company behind it (spin-out reported in January 2026 by [TechCrunch](https://techcrunch.com/2026/01/21/sources-project-sglang-spins-out-as-radixark-with-400m-valuation-as-inference-market-explodes); public launch in May 2026, [RadixArk press release](https://www.businesswire.com/news/home/20260505077157/en/)) | NVIDIA, AMD, Intel Xeon, TPU and Ascend | Continuous batching, RadixAttention prefix caching, chunked prefill, EAGLE and MTP speculation, structured output, expert parallelism with DeepEP, P/D disaggregation |
| TensorRT-LLM | Developed by NVIDIA; PyTorch is the only execution backend after the removal of the TensorRT engine backend ([TensorRT-LLM migration guide](https://nvidia.github.io/TensorRT-LLM/latest/legacy/tensorrt-backend-removal.html)) | NVIDIA | In-flight (continuous) batching, paged KV cache, speculative decoding (EAGLE-3, MTP, n-gram, draft model), disaggregated serving, wide expert parallelism |
| llama.cpp server | ggml.ai, part of Hugging Face since February 2026 ([Hugging Face blog, 20 February 2026](https://huggingface.co/blog/ggml-joins-hf)) | CPU, Apple silicon, consumer GPUs; quantized GGUF models | Continuous batching, prompt-cache reuse, speculative decoding, grammar-constrained output, multiple LoRA adapters |
| Text Generation Inference (TGI) | Maintenance mode ([TGI documentation](https://huggingface.co/docs/text-generation-inference/index)); repository archived on 21 March 2026 ([GitHub](https://github.com/huggingface/text-generation-inference)) | – | Hugging Face recommends vLLM and SGLang, and local engines such as llama.cpp and MLX ([TGI documentation](https://huggingface.co/docs/text-generation-inference/index)) |
