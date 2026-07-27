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

## Run it

Open the notebook in Google Colab, set Runtime > Change runtime type > T4 GPU, and run top to bottom. The first cell pins the stack, so a cold runtime reproduces the book's numbers. It runs on CPU too, but each decode takes 30 to 60 seconds instead of 3 to 8.

Runtime: about 12 minutes on a Colab Free T4, about 25 minutes on a CPU laptop.
