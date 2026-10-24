# Chapter 5 — GPU Architecture and Hardware-Aware Decisions

Companion notebook for Chapter 5. It makes the chapter's claims measurable on a real GPU. It runs on a Colab T4 or any recent NVIDIA GPU.

## Notebook

- [`chapter05_companion.ipynb`](chapter05_companion.ipynb)

## What you measure

1. Bandwidth vs compute: a copy is bound by memory, a large matmul is bound by arithmetic.
2. Coalescing: contiguous access is far faster than strided access for the same byte count.
3. Arithmetic intensity and the roofline: where the three RAG stages fall, computed from real shapes.
4. Mixed precision (FP16): the honest result. FP16 speeds up the compute-bound prefill matmul and does little for memory-bound decode.
5. Figure regeneration: the code that reproduces the chapter's roofline figure.

The notebook closes with exercises that sweep copy size against bandwidth, stride against slowdown, and place your own model's shapes on the roofline.

## Versions

Tested stack (Chapters 1–5 and 7): Python 3.11, PyTorch 2.14.1 (CUDA 13.0), transformers 5.20.2, sentence-transformers 6.2.0. The book's measurements come from an NVIDIA A100 80GB (reference), an H100 and a T4; the case-study data is in [`case_study/reference_results/`](../case_study/reference_results/). Colab ships its own PyTorch build, so the notebooks do not reinstall torch.

## Run it

Open the notebook in Google Colab, set Runtime > Change runtime type > T4 GPU, and run top to bottom. A GPU runtime is required.
