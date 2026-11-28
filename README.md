# High-Performance Machine Learning

Companion code for *High-Performance Machine Learning* (Packt) by **Dr. Kaoutar El Maghraoui**.

A model that is accurate but too slow does not ship. Neither does one that is accurate but too expensive to serve, or one that crashes when traffic doubles. The book is built around that observation. It treats performance as a design concern from the first commit, and it walks through the techniques that actually move the needle: compilation, custom kernels, quantization, continuous batching, KV cache management, distributed training, accelerator portability.

The companion site, <https://kaoutar55.github.io/High-Performance-Machine-Learning/>, shows each chapter's notes and notebooks as readable pages, with an experiment map of the whole book and links back to the code here.

This repo holds runnable notebooks for every chapter. The same RAG case study runs through all of them: a Gemma-2B-IT generator in FP16, a BGE-M3 embedder and a FAISS index. The book measures it on an NVIDIA A100 80GB (the reference platform), an H100 and a T4, and sets every performance target relative to the measured baseline on the same GPU. On the A100 the naïve pipeline answers a query in 379 ms at the median, and token-by-token decode takes 89% of the sum of the stage medians (Chapter 1, Table 1.3).

## Platforms

The case study uses the same two models on every GPU. Absolute numbers change with the GPU and the host; the ratios and the principles the book teaches hold on all three.

| Platform | Role in the book | Memory | Generator | Embedder | Naïve p50 latency (measured) |
|---|---|---|---|---|---|
| NVIDIA A100 80GB (SXM) | Reference: quoted first in every table | 80 GB HBM2e | Gemma-2B-IT, FP16 | BGE-M3 (568M), FP16 | 379 ms |
| NVIDIA H100 80GB | Current generation | 80 GB HBM3 | Gemma-2B-IT, FP16 | BGE-M3 (568M), FP16 | 189 ms |
| NVIDIA T4 16GB | Budget path (free Colab tier) | 16 GB GDDR6 | Gemma-2B-IT, FP16 | BGE-M3 (568M), FP16 | 673 ms |

The latencies are the medians of five independent runs of the Chapter 2 lab (100 requests each, PyTorch 2.14.1, Modal, 23 October 2026), as printed in Chapter 1, Table 1.3; the data is in [`case_study/reference_results/x5_2026-10-23/master_tables.md`](case_study/reference_results/x5_2026-10-23/master_tables.md).

FP16 is used throughout because the T4 has no BF16 Tensor Core path. Chapter 6 has its own pin: its notebook (`chapter06_companion_2p14.ipynb`) is pinned to PyTorch 2.14.1 (CUDA 13.0 build) and Triton 3.8.0, and its reference measurements were taken on an NVIDIA H100 80GB HBM3 on 24 November 2026 (see its README). The Chapter 1 lab still contains an older hardware-tier recipe with smaller models for CPU-only machines.

## Case study and principles

[`case_study/`](case_study/README.md) re-measures the case study on Modal on all three GPUs in one command and checks the book's principles against the measurements. It separates the claims the book teaches (`principles.py`, each with the chapters that teach it and its sources) from the swappable settings (`experiment_config.py`: GPUs, models, sizes, package versions). The measured data every chapter quotes is in [`case_study/reference_results/`](case_study/reference_results/README.md). The list of the book's 51 Performance Engineering Principles is in [`docs/principles.md`](docs/principles.md).

## Repository layout

```
.
├── README.md                     this file
├── LICENSE                       MIT
├── chapter01/
│   ├── README.md
│   └── chapter01_lab.ipynb
├── chapter02/
│   ├── README.md
│   └── chapter02_lab.ipynb
├── chapter03/
│   ├── README.md
│   ├── chapter03_exercises.ipynb
│   └── chapter03_solutions.ipynb
├── chapter04/
│   ├── README.md
│   └── chapter04_companion.ipynb
├── chapter05/
│   ├── README.md
│   └── chapter05_companion.ipynb
├── chapter06/                    Triton kernels (pinned PyTorch 2.14.1 + Triton 3.8.0)
├── chapter07/                    distributed training (analytical cells + measured 2x A10G runs)
├── chapter08/                    inference serving (analytical scripts + vLLM case study on Modal)
├── chapter09/                    quantization and sparsity (teaching scripts + llm-compressor/vLLM case study on Modal)
├── chapter10/                    model adaptation (PEFT, distillation) case study
├── chapter11/                    accelerators and compilers case study (GPU, TPU, ONNX, JAX)
├── chapter12/                    multimodal agentic RAG case study
├── case_study/                   multi-GPU re-measurement, principles, reference results
├── tools/drift_check/            monthly check that the code and numbers still hold
└── ...                           one folder per chapter as the book is released
```

Each chapter folder has its own README, and Chapters 1 to 8 have a Colab-ready notebook; Chapters 9 to 12 provide scripts and case-study runners. Notebook section headings name the chapter section they accompany, so you can read with the notebook open beside you. Each README states the tested stack: Python 3.11, PyTorch 2.14.1 (CUDA 13.0), transformers 5.20.2 and sentence-transformers 6.2.0 for Chapters 1–5 and 7, and PyTorch 2.14.1 + Triton 3.8.0 for Chapter 6. Chapters 8 and 9 run their case studies on Modal with vLLM 0.32.0, and Chapter 9 builds its checkpoints with llm-compressor 0.14 in a separate image; their teaching scripts run on a CPU.

## Quick start

### Option 1: Google Colab

1. Open the notebook on GitHub and click **Open in Colab**.
2. *Runtime → Change runtime type → Hardware accelerator*: a T4 works for Chapters 1 to 5 and 7; Chapter 6 needs a GPU of compute capability 8.0 or newer (Triton does not support the T4), and an A100 or H100 matches the book's reference numbers.
3. Run the cells top to bottom.

That's the path most readers should take.

### Option 2: Local

```bash
git clone https://github.com/PacktPublishing/High-Performance-Machine-Learning.git
cd High-Performance-Machine-Learning

python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install jupyter
jupyter lab
```

Install PyTorch for your CUDA version first (https://pytorch.org/get-started/locally/); each notebook then installs the libraries it needs. Without a CUDA GPU the timings are not comparable with the book's.

## Chapter map

| #  | Title                                              | Notebook              | What you produce                                           |
|----|----------------------------------------------------|-----------------------|------------------------------------------------------------|
| 1  | The Imperative of Performance in Modern AI         | `chapter01/`          | the four factors, hardware tiering, naïve RAG baseline     |
| 2  | A Methodology for Performance Engineering          | `chapter02/`          | profiler runs, Roofline placement, regression test         |
| 3  | PyTorch Internals and Execution Performance        | `chapter03/`          | the Python tax, launch overhead, autograd cost, DataLoader |
| 4  | Optimizing with `torch.compile` and Kernel Fusion  | `chapter04/`          | graph breaks, fused kernels, eager-vs-compiled gap         |
| 5  | GPU Architecture and Hardware-Aware Decisions      | `chapter05/`          | bandwidth vs compute, coalescing, the Roofline model       |
| 6  | Practical High-Performance Kernels with Triton     | `chapter06/`          | Triton kernels: fused softmax, LayerNorm/RMSNorm, tiled matmul, autotuning |
| 7  | Scaling Out: Choosing a Distributed Training Strategy | `chapter07/`       | memory wall, all-reduce cost, pipeline bubbles, DDP and FSDP measured |
| 8  | High-Performance Inference and Serving             | `chapter08/`          | KV cache sizing, paged KV, speculative decoding, load sweeps and goodput on three GPUs |
| 9  | Model Compression: Quantization and Sparsity       | `chapter09/`          | FP8, INT8 and INT4 quantization, 2:4 sparsity, speed and quality on three GPUs |
| 10 | Efficient Model Adaptation and Maintenance | `chapter10/` | LoRA, QLoRA and full fine-tuning: cost, quality, forgetting, serving adapters, distillation |
| 11 | Navigating the Accelerator and Compiler Landscape | `chapter11/` | the same generators on A100, H100, TPU v5e and v6e; parity, determinism, ONNX, sharding |
| 12 | The Mature ML System | `chapter12/` | multimodal agentic RAG: optimization ladder, routing at scale-out, GPU choice, regression gate |

Packt publishes the book in chapter installments through their Early Access Program. New folders show up here as each chapter clears editorial.

## Book editions

Each published edition of the book is a git tag (`edition-1`, `edition-2`, ...). The tag fixes the notebooks, the pinned versions and the reference results that edition quotes. To read the code exactly as printed:

```bash
git fetch --tags
git checkout edition-1
```

The default branch keeps moving with PyTorch, Triton and new GPUs. [`CHANGES_SINCE_PUBLICATION.md`](CHANGES_SINCE_PUBLICATION.md) records behaviour that changed after an edition went to print, and [`RELEASING.md`](RELEASING.md) describes how an edition is cut.

## What's *not* in this repository

Manuscript drafts, figure source files and other working materials are kept outside this repository. Only finished, runnable code lands here.

## Citing this book

```bibtex
@book{elmaghraoui_hpml_2026,
  author    = {El Maghraoui, Kaoutar},
  title     = {High-Performance Machine Learning},
  publisher = {Packt Publishing},
  year      = {2026},
  url       = {https://github.com/PacktPublishing/High-Performance-Machine-Learning}
}
```

## License

Code is MIT, see [LICENSE](LICENSE). The book text and figures are © Kaoutar El Maghraoui / Packt Publishing and are not redistributed here.

## Contact and errata

For code bugs, environment issues, or numerical regressions, file a GitHub issue. For book errata, use Packt's errata process.
