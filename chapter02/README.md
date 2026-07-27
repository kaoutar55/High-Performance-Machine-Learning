# Chapter 2 — A Methodology for Performance Engineering

Lab notebook for Chapter 2. You run the full performance-engineering methodology on the RAG case study, end to end, on a real Colab T4.

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

## Run it

Open the notebook in Google Colab, set Runtime > Change runtime type > T4 GPU, and run top to bottom. It runs on CPU too, but the timings and Roofline placement are only meaningful on a GPU.
