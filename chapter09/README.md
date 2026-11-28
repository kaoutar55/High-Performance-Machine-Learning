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
modal run chapter09/modal_quant.py --what quality        # 3. KL, SQuAD answers and lm-eval vs baseline, on the H100
modal run chapter09/modal_quant.py --what perf           # 4. memory, latency and capacity on all three GPUs
modal run chapter09/modal_quant.py --what prefill,long   # 5. prefill-heavy and long-context studies
modal run chapter09/modal_quant.py --what squad          # adds SQuAD scores to quality results from older runs
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
- Red Hat's W4A16 Llama checkpoint uses dynamic act-order (desc_act=True), which vLLM 0.32 rejects, so the W4A16
  speed runs use the chapter's own GPTQ checkpoint (`w4a16_gptq_rag`, static act-order).
- Gemma with an FP8 KV cache is not run on the H100: FlashAttention-3 accepts an FP8 KV cache only for BF16 models,
  and Gemma runs in FP16 throughout. On the T4 it fails at startup, which the perf study records.
- The two Llama FP8-KV variants, `fp8_dynamic_kv_scale1` and `fp8_dynamic_kv_cal`, come from the same FP8_DYNAMIC
  recipe and differ only in the KV scales (1.0 versus calibrated), so their comparison isolates the scales.
- Quality also scores each answer against the SQuAD reference answers of its question (answer contained in the
  first line, and token F1), paired with the 16-bit model on the same 200 held-out prompts, with flips.
- Every A100 result comes from the SXM4 card; the scripts reject a PCIe A100 and retry.

## Online resources

The print chapter refers to the material below. Reference numbers in brackets are those of the chapter's reference
list. The items that cite case-study results describe the runs that the chapter reports: vLLM 0.32.0 and
llm-compressor 0.14 on the A100 80GB (SXM4), H100 and T4 of Chapter 8, and the rebuilt 2:4 checkpoints and the
comparison of pruning criteria measured on one H100 80GB on 25 November 2026.

- Knowledge check answers: [`knowledge_check_solutions.md`](knowledge_check_solutions.md).
- The list of the book's 51 Performance Engineering Principles, with the chapter that states each one:
  [`docs/principles.md`](../docs/principles.md).

### GPTQ act-order and vLLM 0.32

The original GPTQ act-order (desc_act=True) reorders the group scales and slows the kernels. vLLM 0.32 rejects
checkpoints that use it and accepts the static act-order variant that llm-compressor produces. For this reason the
case study builds its own W4A16 checkpoints with llm-compressor in place of Red Hat's published W4A16 checkpoint.

### FP8 KV cache backends in vLLM 0.32

In vLLM 0.32 the cache format is chosen with --kv-cache-dtype, and what it does depends on the GPU [40]:

- H100: fp8 runs on the FlashAttention-3 backend, which also quantizes the query and computes attention in FP8.
  The cache is halved and the attention arithmetic runs at the FP8 rate.
- A100: FlashAttention-2 does not accept an FP8 cache, so vLLM uses the FlashInfer backend, which stores K and V in
  FP8 and converts them to 16 bits inside the kernel. The cache is halved; the attention arithmetic stays in 16 bits.
- T4: no attention backend in vLLM 0.32 accepts an FP8 cache on compute capability 7.5, and the server fails at
  startup with an error that suggests --kv-cache-dtype float16 (recorded in
  [`results/T4_gemma.errors.json`](results/T4_gemma.errors.json)).

### 2:4 sparsity support history (as of the case-study runs, October 2026)

- vLLM removed its 2:4 kernels in 2026: Sparse-Marlin after version 0.15 and CUTLASS after 0.18 [51]. vLLM 0.32
  refuses checkpoints with a sparsity configuration [52].
- torchao removed its sparsity module in August 2026 and kept FP8 2:4 tensors in its quantization workflows
  [53][54].
- PyTorch keeps torch.sparse.to_sparse_semi_structured, which runs a pruned 16-bit or INT8 weight with cuSPARSELt
  or CUTLASS (`sparse_gemm.py` in this folder uses it). In PyTorch 2.14.1 the CUTLASS backend supports only compute
  capability 8.x.
- TensorRT-LLM's TensorRT engine backend offered 2:4 weight sparsity through the `--weight_sparsity` option of
  `trtllm-build` ([TensorRT-LLM 0.21.0 documentation](https://nvidia.github.io/TensorRT-LLM/0.21.0/commands/trtllm-build.html)).
  That backend has since been removed, and PyTorch is the only execution backend
  ([TensorRT-LLM migration guide](https://nvidia.github.io/TensorRT-LLM/latest/legacy/tensorrt-backend-removal.html)).
- On the B300, the sparse FP4 rate is 1.33 times the dense rate [55].

### Build history of the 2:4 checkpoints (added 28 November 2026)

The print chapter reports the final 2:4 builds. This section records how they were reached.

- Gemma-2B. With the recipe first used for the chapter, the Wanda and SparseGPT builds of Gemma-2B gave WikiText
  perplexities of 8.9 × 10¹⁰ and 1.6 × 10⁶ in the same run, higher than magnitude pruning. That order contradicted
  the method papers and the `prune24.py` result, so the saved checkpoints were checked tensor by tensor for the
  fraction of zeros. llm-compressor 0.14.0.1, called with `targets=["Linear"]` and `ignore=["lm_head"]`, had also
  pruned the output projection (`lm_head`) to 2:4. Gemma-2B shares this matrix with its input embedding, so half of
  the values of every token embedding were zero. Listing the seven decoder projections as the targets kept the
  embedding dense and gave the results in the chapter (`gemma24_rerun.py`,
  [`results/gemma_24_rerun_2026-11-25.json`](results/gemma_24_rerun_2026-11-25.json)). The re-run evaluated every
  model in BF16, the precision in which the checkpoints were pruned, to test whether evaluation in FP16 had caused
  the high perplexities of the first builds.
- Llama-3.1-8B. The first SparseGPT 2:4 build had also pruned the output layer (`lm_head`), although the recipe
  excluded it. Rebuilt with the output layer dense, it lost 73.8 GSM8K points against 74.8 for the first build. The
  2:4 row of Table 9.11 is the rebuilt checkpoint, measured on 25 November 2026 on one H100 80GB with vLLM 0.32.0,
  with the same prompts, tasks and BF16 baseline (`llama24_rerun.py`,
  [`results/llama_24_rerun_2026-11-25.json`](results/llama_24_rerun_2026-11-25.json)). Between the two builds, every
  task changed by less than two of the paired standard errors in Table 9.11.
