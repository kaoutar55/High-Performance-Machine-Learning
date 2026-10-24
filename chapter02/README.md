# Chapter 2 — A Methodology for Performance Engineering

Lab notebook for Chapter 2. You run the full performance-engineering methodology on the RAG case study, end to end, on a real GPU (a Colab T4 works).

## Notebook

- [`chapter02_lab.ipynb`](chapter02_lab.ipynb)

## What you produce

1. A latency distribution (p50 / p95 / p99) over 100 queries.
2. A stage-level waterfall.
3. A per-kernel PyTorch profiler ranking.
4. A Chrome trace you can open at https://ui.perfetto.dev.
5. NVTX-annotated code that produces a trace readable in Perfetto or Nsight Systems.
6. A Roofline analysis with measured arithmetic intensity.
7. A regression test that fails if decode latency regresses by more than 15%.

## Versions

Tested stack (Chapters 1–5 and 7): Python 3.11, PyTorch 2.14.1 (CUDA 13.0), transformers 5.20.2, sentence-transformers 6.2.0. The book's measurements come from an NVIDIA A100 80GB (reference), an H100 and a T4; the case-study data is in [`case_study/reference_results/`](../case_study/reference_results/). Colab ships its own PyTorch build, so the notebooks do not reinstall torch.

## Run it

Open the notebook in Google Colab, set Runtime > Change runtime type > T4 GPU, and run top to bottom. It runs on CPU too, but the timings and Roofline placement are only meaningful on a GPU.
