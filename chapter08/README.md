# Chapter 8 — High-Performance Inference and Serving

Companion code for Chapter 8. The analytical scripts run on any machine; the case study serves the RAG pipeline's generator (Gemma-2B-IT, FP16) with vLLM and measures it under load on an A100 80GB, an H100 and a T4.

## What you compute and measure

1. KV cache bytes per token and token capacity for any Hugging Face model (`kv_calc.py`).
2. The bookkeeping of a paged KV cache: block tables, reference counts, copy-on-write and prefix reuse (`paged_kv.py`).
3. Attention computed tile by tile with an online softmax, and the split-KV combine of Flash-Decoding (`tiled_attention.py`).
4. Expected tokens per target pass and the modeled speedup of speculative decoding (`spec_math.py`).
5. The case study: single-request latency, a load sweep with goodput, prefix caching on and off, the prefill token budget, and n-gram speculative decoding (`modal_serving.py`).

## Run it

```bash
python chapter08/paged_kv.py
python chapter08/spec_math.py
python chapter08/tiled_attention.py                     # needs PyTorch (CPU is fine)
python chapter08/kv_calc.py google/gemma-2b-it --weight-gb 5.0

# case study on Modal (needs a Modal secret named "huggingface" with access to google/gemma-2b-it)
modal run chapter08/modal_serving.py                    # all GPUs, all studies
modal run chapter08/modal_serving.py --gpus T4 --studies single,sweep
python chapter08/summarize_results.py                   # tables and results/sweep.png
```

To run the case study on your own GPU instead, build the workload once and point the load generator at a running server:

```bash
python chapter08/build_prompts.py --out-dir data --num 1000
vllm serve google/gemma-2b-it --dtype float16 --max-model-len 2048
python chapter08/loadgen.py --model google/gemma-2b-it --prompts data/prompts_single.jsonl \
    --rate 4 --num 200 --max-tokens 128 --slo-ttft-ms 500 --slo-tpot-ms 50
```

## Files

| File | What it does |
|---|---|
| `kv_calc.py` | KV bytes per token from a model config; tokens that fit in the memory left after the weights |
| `paged_kv.py` | Teaching model of a paged KV cache manager (no tensors, runs anywhere) |
| `tiled_attention.py` | Online-softmax attention and split-KV decoding, checked against the reference |
| `spec_math.py` | Speculative decoding: expected tokens per pass and speedup against acceptance rate and draft length |
| `build_prompts.py` | Builds the RAG serving workload (1,000 distinct SQuAD questions, BGE-M3 + FAISS top-5 over the Chapter 2 corpus) and a follow-up turn for each |
| `loadgen.py` | Open-loop Poisson load generator for OpenAI-compatible servers; TTFT, ITL, TPOT, E2E percentiles and goodput |
| `modal_serving.py` | Runs the case-study studies on Modal, one fresh `vllm serve` per configuration; writes `results/<gpu>.json` |
| `summarize_results.py` | Prints the case-study tables and plots the load sweep |

## Versions

The case study pins vLLM 0.32.0 in its own image (vLLM pins its own PyTorch build). Set `CH08_VLLM` to try another version, and record the version with your results: serving engines change their defaults between releases.
