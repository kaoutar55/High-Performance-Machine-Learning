# High-Performance Machine Learning

Companion site for *High-Performance Machine Learning* (Packt) by [**Dr. Kaoutar El Maghraoui**](about.md). It collects the
supplementary material for each chapter: the chapter notes, the notebooks rendered as readable pages, the measured
results, and links to the code in the [GitHub repository](https://github.com/kaoutar55/High-Performance-Machine-Learning).

## The running example

One system runs through the whole book: a retrieval-augmented generation (RAG) pipeline. BGE-M3 embeds the documents
into a FAISS index, a query retrieves the top passages, and a generator writes the answer. The generator grows from
Gemma-2B-IT (Chapters 1–9) to Llama-3.1-8B-Instruct (Chapters 8–10) and Qwen3-VL-8B-Instruct (Chapters 10 and 12),
where the system becomes a multimodal agent that retrieves page images as well as text. Each chapter measures or
changes one part of it. The [experiment map](experiment-map.md) shows which part, with the main measurements.

## Chapters

| # | Chapter | Material |
|---|---|---|
| 1 | The Imperative of Performance in Modern AI | [Notes and lab](book/chapter01.md) |
| 2 | A Methodology for Performance Engineering | [Notes and lab](book/chapter02.md) |
| 3 | PyTorch Internals and Execution Performance | [Notes, exercises and solutions](book/chapter03.md) |
| 4 | Optimizing with `torch.compile` and Kernel Fusion | [Notes and notebook](book/chapter04.md) |
| 5 | GPU Architecture and Hardware-Aware Decisions | [Notes and notebook](book/chapter05.md) |
| 6 | Practical High-Performance Kernels with Triton | [Notes and notebooks](book/chapter06.md) |
| 7 | Scaling Out: Choosing a Distributed Training Strategy | [Notes, scripts and notebook](book/chapter07.md) |
| 8 | High-Performance Inference and Serving | [Notes, scripts and notebook](book/chapter08.md) |
| 9 | Model Compression: Quantization and Sparsity | [Notes and scripts](book/chapter09.md) |
| 10 | Efficient Model Adaptation and Maintenance | [Notes, scripts and results](book/chapter10.md) |
| 11 | Navigating the Accelerator and Compiler Landscape | [Chapter page](book/chapter11.md) |
| 12 | The Mature ML System | [Notes, scripts and results](book/chapter12.md) |

The [case study and principles](book/case_study.md) page describes how the book's measurements are repeated on
several GPUs and checked against the principles the book states. The [list of the 51 Performance Engineering Principles](principles.md) gives each principle with
the chapter that states it.

## Hardware

The book measures the case study on an NVIDIA A100 80GB (the reference platform), an H100 80GB and a T4 16GB.
Chapters 10 and 12 also use multi-GPU H100 runs, and Chapter 11 adds Google Cloud TPUs. Absolute numbers change with
the GPU and the host; each chapter page states the stack its results were measured with.

## Book editions

Each published edition is a git tag (`edition-1`, ...) that fixes the code, the pinned versions and the measured
results that edition quotes. This site follows the default branch.
