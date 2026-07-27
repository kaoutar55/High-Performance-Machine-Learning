# Chapter 3 — PyTorch Internals and Execution Performance

Seven programming exercises, one per major section of the chapter. Every one is about measuring something, because that is the chapter's whole argument: profile, don't guess.

## Notebooks

- [`chapter03_exercises.ipynb`](chapter03_exercises.ipynb) — starter cells with `TODO`s to fill in.
- [`chapter03_solutions.ipynb`](chapter03_solutions.ipynb) — complete, runnable reference solutions.

## The exercises

1. The Python tax: sum a million floats four ways and disassemble one `+`.
2. Where launch overhead dominates: watch per-op dispatch cost hide as the batch grows.
3. What autograd costs at inference: latency and peak memory with autograd on, `no_grad`, `inference_mode`.
4. Sweeping the DataLoader: throughput vs `num_workers` and `pin_memory`.
5. Hunt the anti-pattern: fix a correct-but-slow training step.
6. Benchmark without lying to yourself: the difference between timing with and without a CUDA sync.
7. Capstone: optimize the RAG embedding stage.

## Run it

Runs on Google Colab free tier. Exercises 3.3, 3.6, and 3.7 want a GPU (Runtime > Change runtime type > T4); the rest run on CPU. `torch` and `numpy` are preinstalled on Colab. Numbers vary by hardware; the ratios are what matter.
