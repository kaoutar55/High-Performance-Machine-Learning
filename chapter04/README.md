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
