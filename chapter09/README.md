# Chapter 9 — Model Compression: Quantization and Sparsity

Companion code for Chapter 9. The teaching scripts run on a CPU; the case study quantizes and prunes the RAG
pipeline's generators (Gemma-2B-IT and Llama-3.1-8B-Instruct), serves every variant with vLLM 0.32.0 on an
A100 80GB (SXM4), an H100 and a T4, and measures what each one changes in memory, speed, capacity and quality.

## Teaching scripts

| Script | What it shows |
|---|---|
| `quant_basics.py` | Scale and zero point; per-tensor, per-channel, per-group and per-token granularity; INT8, INT4 and FP8 error on a layer with outlier channels |
| `smoothquant_demo.py` | Moving activation outliers into the weights before W8A8 quantization |
| `prune24.py` | Magnitude vs Wanda pruning, unstructured vs 2:4, on the same layer |
| `decode_bytes.py` | Bytes read per decode step and the batch-1 latency floor for each format and GPU |
| `sparse_gemm.py` | Dense vs 2:4 sparse GEMM timing on the GPU for Llama-3.1-8B's layer shapes |

```bash
python chapter09/decode_bytes.py
python chapter09/quant_basics.py           # needs PyTorch (CPU is fine)
python chapter09/smoothquant_demo.py
python chapter09/prune24.py
python chapter09/sparse_gemm.py            # needs an A100 or H100
```

## Case study on Modal

Needs the Modal secret `huggingface` (a token that can read `google/gemma-2b-it` and
`meta-llama/Llama-3.1-8B-Instruct`) and the Chapter 8 prompt volume (`modal run chapter08/modal_serving.py
--studies single` builds it once).

```bash
modal run chapter09/quantize.py                          # 1. build the compressed checkpoints (A100, ~2-3 h)
modal run chapter09/sparse_gemm.py                       # 2. 2:4 GEMM benchmark and the CPU listings
modal run chapter09/modal_quant.py --what quality        # 3. KL vs baseline and lm-eval, on the H100
modal run chapter09/modal_quant.py --what perf           # 4. memory, latency and capacity on all three GPUs
modal run chapter09/modal_quant.py --what prefill,long   # 5. prefill-heavy and long-context studies
python chapter09/summarize_results.py                    # tables for the chapter
```

What each variant is, and which GPU runs it, is listed at the top of `modal_quant.py`. Notes:

- vLLM 0.32 and llm-compressor 0.14 pin different torch, transformers and compressed-tensors versions, so they
  run in separate images; checkpoints move between them through the `hpml-ch09-ckpt` volume.
- vLLM 0.32 no longer serves 2:4 sparse checkpoints (removed in 0.19). Pruned models are saved as dense weights
  and evaluated for quality only; `sparse_gemm.py` measures the kernel speedup separately.
- FP8 weight-and-activation compute needs compute capability 8.9 or newer. On the A100 and T4, FP8 checkpoints
  run weight-only (FP8 weights, 16-bit math). `--kv-cache-dtype fp8` works on the H100 (FlashAttention-3) and
  the A100 (FlashInfer), and fails on the T4.
- Llama-3.1-8B does not fit on the 16 GB T4 in 16 bits; there it runs only quantized.
- Every A100 result comes from the SXM4 card; the scripts reject a PCIe A100 and retry.
