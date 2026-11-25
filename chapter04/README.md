# Chapter 4 — Optimizing with torch.compile and Kernel Fusion

Companion notebook for Chapter 4. It makes the chapter's claims measurable on a real GPU, using GPT-2 as the model under test.

## Notebook

- [`chapter04_companion.ipynb`](chapter04_companion.ipynb)

## What you measure

1. Graph break diagnosis: use `torch._dynamo.explain` to find and fix breaks in a real model.
2. Profiler comparison: profile eager vs compiled GPT-2 and see fusion in the kernel table.
3. Custom operator registration: register a custom op with `torch.library` so it compiles cleanly.
4. Warm-up timing: cold compile vs warm execution, and the effect of regional compilation.

The notebook closes with five exercises, from confirming that a `torch.where` branch does not break the graph to compiling with `fullgraph=True` and tracing any break to its source.

## Versions

Tested stack (Chapters 1–5 and 7): Python 3.11, PyTorch 2.14.1 (CUDA 13.0), transformers 5.20.2, sentence-transformers 6.2.0. The book's measurements come from an NVIDIA A100 80GB (reference), an H100 and a T4; the case-study data is in [`case_study/reference_results/`](../case_study/reference_results/). Colab ships its own PyTorch build, so the notebooks do not reinstall torch.

## Run it

Open the notebook in Google Colab, set Runtime > Change runtime type > T4 GPU, and run top to bottom.

## Online resources

The print chapter refers to the material below. It is version-specific: check it against the PyTorch and transformers
releases you run.

### CUDA graphs with a hand-compiled forward: failure on PyTorch 2.14.1 and 2.15.0

With `mode="reduce-overhead"`, which adds CUDA graphs, the hand-compiled forward of Gemma-2B-IT with a static KV cache
crashed the process on all three GPUs with PyTorch 2.14.1 and transformers 5.20.2 (Modal, 23 October 2026), and again
in a re-test on 23 November 2026 with both PyTorch 2.14.1/transformers 5.20.2 and PyTorch 2.15.0/transformers 5.24.0
(Modal T4, Modal A100 80GB, H100; 30 of 30 runs ended with a segmentation fault). The static cache in these
transformers releases updates a length counter in place (`cumulative_length.add_`) inside the compiled region, and the
crash trace points to that update.

Workaround measured on 23 November 2026: call the compiled forward only for the one-token decode steps and run the
prefill in eager mode. On the A100 this decoded at 6.1 ms per token, against 5.9 ms for transformers' automatic
compilation (pattern 2) in the same run; on the H100 3.57 to 3.68 ms (pattern 2: 3.49 to 3.55 ms); on the T4 23.7 ms
(pattern 2: 23.5 to 23.8 ms). transformers' automatic compilation splits prefill and decode in the same way.

With Inductor graph partitioning switched off (`graph_partition=False`), the run completes, but Inductor skips CUDA
graphs for the whole forward because of mutated inputs (`cumulative_length.add_`, `keys.index_copy_`), so it runs at
the speed of `mode="default"`.

Results: [`results/reduce_overhead_retest_2026-11-23.json`](results/reduce_overhead_retest_2026-11-23.json); the crash
traces are the `results/reduce_overhead_trace_*_2026-11-23.txt` files.

### CUDA graph capture and torch.inference_mode

The re-test script [`scripts/reduce_overhead_retest.py`](scripts/reduce_overhead_retest.py) runs every variant inside
`torch.inference_mode`. With PyTorch 2.14.1 and 2.15.0 (23 November 2026), transformers' automatic compilation
(pattern 2), which captures the decode step in CUDA graphs, ran to completion under `torch.inference_mode` on the T4,
A100 and H100 ([`results/reduce_overhead_retest_2026-11-23.json`](results/reduce_overhead_retest_2026-11-23.json)).
If CUDA graph capture fails inside `torch.inference_mode` with another PyTorch release, `torch.no_grad` also disables
gradient tracking and is the configuration to test next.

### Knowledge check, question 4: answer note

Answer: (c). reduce-overhead targets the per-step launch overhead of batch-1 decode with CUDA graphs. In the case
study, transformers' automatic compilation used it for the decode step and ran; compiling `model.forward` by hand with
it crashed with PyTorch 2.14.1 and 2.15.0 (see the first item above). Principle #10 still applies: start with the
default mode, then try reduce-overhead where the work is launch-bound.
