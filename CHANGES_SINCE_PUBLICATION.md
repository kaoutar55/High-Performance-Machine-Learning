# Changes since publication

PyTorch, Triton, Hugging Face libraries and GPUs keep changing after a book goes to print. This page
records behaviour that differs from what an edition describes, or that a reader will meet when running
the code on a newer stack. Each entry says what changed, which versions show it, what it means for the
book's text, and what to do.

The principles the book teaches are checked automatically on every run of the case study
(`case_study/principles.py`) and once a month by the drift check (`tools/drift_check/`). A drift-check
failure is the usual reason for a new entry here.

## Template

```
### <short title>  (<date found>)
- Affects: Chapter N, <section or figure/table>
- Versions: <library and version range, GPU>
- What changed: <one or two sentences>
- What to do: <workaround, or "nothing; the text still holds">
- Evidence: <issue, release note, or results file>
```

## Entries

### CUDA-graph compilation of the decode step crashes with the static KV cache  (2026-10-23)
- Affects: Chapter 4, compiled-decode case study
- Versions: PyTorch 2.14.1 + transformers 5.20.2, on A100, H100 and T4
- What changed: compiling `model.forward` with `torch.compile(mode="reduce-overhead")` and decoding
  with `cache_implementation="static"` crashes the process (SIGSEGV). The static cache updates a counter
  in place (`cumulative_length.add_` in `transformers/cache_utils.py`) inside the captured CUDA graph.
- What to do: use `mode="default"` or transformers' own compilation (below). Both were measured and work.
- Evidence: `case_study/reference_results/run2_2026-10-23/*.json`, `compiled_decode.runs`

### transformers compiles `generate()` automatically when the cache is static  (2026-10-23)
- Affects: Chapter 4
- Versions: transformers 5.x (GenerationConfig has `disable_compile`)
- What changed: `generate(..., cache_implementation="static")` compiles the decode step without any
  call to `torch.compile`. An "eager static cache" measurement is compiled unless `disable_compile=True`
  is passed.
- What to do: pass `disable_compile=True` for an eager baseline. The case study measures both.
- Evidence: `dynamo_graphs_compiled` in the reference results (0 with the switch, 1 without)

### Triton no longer supports Turing GPUs (T4)  (2026-10-23)
- Affects: Chapter 6
- Versions: Triton 3.3 and later
- What changed: Triton requires compute capability 8.0 or newer. The T4 (7.5) is not supported.
- What to do: on a T4, install the chapter's pinned pair: `pip install torch==2.6.0 triton==3.2.0`.
- Evidence: Triton README and the 3.3 release notes

### SDPA chooses cuDNN attention on H100  (2026-10-23)
- Affects: Chapters 2, 3 and 5 (attention kernel names in profiles)
- Versions: PyTorch 2.5 and later, Hopper and newer GPUs
- What changed: `scaled_dot_product_attention` dispatches to cuDNN fused attention by default on H100.
  The profiler shows `aten::_cudnn_attention_forward`. On A100 and T4 the BGE-M3 embedder runs the
  memory-efficient (CUTLASS) kernel, because FlashAttention does not accept its padding mask and the T4
  cannot run FlashAttention.
- What to do: read the kernel name from the profiler; the book's tables report it per GPU.
- Evidence: PyTorch 2.5 release notes; `embedder.attention_backend` in the reference results

### Overhead-bound numbers vary with the host machine on Modal  (2026-10-23)
- Affects: Chapters 2–5 (batch-1 decode and batch-1 encoding)
- Versions: any; seen on Modal A100 and H100
- What changed: two runs of the same code and stack gave 9.0 and 17.5 ms per decode token on an A100.
  Compute- and bandwidth-bound numbers (prefill, large-batch encoding, T4 decode) moved about 5%.
- What to do: report the median and range over independent runs (`--repeats 5`); quote ratios and
  principle verdicts with more confidence than absolute overhead-bound times.
- Evidence: `case_study/reference_results/README.md`
