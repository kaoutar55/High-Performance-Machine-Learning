# Chapter 1 — The Imperative of Performance in Modern AI

Lab notebook for Chapter 1. You measure the ideas the chapter argues for on hardware you actually have.

## Notebook

- [`chapter01_lab.ipynb`](chapter01_lab.ipynb)

## What you produce

1. An auto-detected hardware tier (consumer CPU, consumer GPU 16 GB, datacenter 40 GB, datacenter 80 GB+).
2. A measured latency-throughput trade-off curve with the three operating zones shaded in.
3. A working naïve RAG baseline: embedding plus FAISS plus a small instruct LM.
4. An end-to-end latency waterfall for a single query.
5. A short accuracy-performance gap demo: same model, same answer, very different cost per query.
6. Answers to the end-of-chapter Knowledge Check.

Save the end-to-end latency you measure. Chapter 2's regression test compares against it.

## Versions

Tested stack (Chapters 1–5 and 7): Python 3.11, PyTorch 2.14.1 (CUDA 13.0), transformers 5.20.2, sentence-transformers 6.2.0. The book's measurements come from an NVIDIA A100 80GB (reference), an H100 and a T4; the case-study data is in [`case_study/reference_results/`](../case_study/reference_results/). Colab ships its own PyTorch build, so the notebooks do not reinstall torch.

## Run it

Open the notebook in Google Colab, set Runtime > Change runtime type > T4 GPU, and run top to bottom. The first cell installs the tested library versions. It runs on CPU too, but decoding is far slower.

Runtime, measured on Modal in November 2026 with the tested stack: 73 s on a T4 and 252 s on a 2-vCPU cloud CPU, plus the model downloads on a fresh machine (results/lab_corpus_check_2026-11-23.json). Timings on your machine will differ, most of all on CPU, where the host processor sets the speed.
