# High-Performance Machine Learning

Companion code for *High-Performance Machine Learning* (Packt) by **Dr. Kaoutar El Maghraoui**.

A model that is accurate but too slow does not ship. Neither does one that is accurate but too expensive to serve, or one that crashes when traffic doubles. The book is built around that observation. It treats performance as a design concern from the first commit, and it walks through the techniques that actually move the needle: compilation, custom kernels, quantization, continuous batching, KV cache management, distributed training, accelerator portability.

This repo holds runnable notebooks for every chapter. The same RAG case study runs through all of them: a Gemma-2B-IT generator in FP16, a BGE-M3 embedder and a FAISS index. The book measures it on an NVIDIA A100 80GB (the reference platform), an H100 and a T4, and sets every performance target relative to the measured baseline on the same GPU. On the A100 the naïve pipeline answers a query in about 380 ms at the median, and token-by-token decode takes almost 90% of that time.

## Platforms

The case study uses the same two models on every GPU. Absolute numbers change with the GPU and the host; the ratios and the principles the book teaches hold on all three.

| Platform | Role in the book | Memory | Generator | Embedder | Naïve p50 latency (measured) |
|---|---|---|---|---|---|
| NVIDIA A100 80GB (SXM) | Reference: quoted first in every table | 80 GB HBM2e | Gemma-2B-IT, FP16 | BGE-M3 (568M), FP16 | ~380 ms |
| NVIDIA H100 80GB | Current generation | 80 GB HBM3 | Gemma-2B-IT, FP16 | BGE-M3 (568M), FP16 | ~220 ms |
| NVIDIA T4 16GB | Budget path (free Colab tier) | 16 GB GDDR6 | Gemma-2B-IT, FP16 | BGE-M3 (568M), FP16 | ~590 ms |

FP16 is used throughout because the T4 has no BF16 Tensor Core path. Chapter 6 is the exception to the stack: it is pinned to PyTorch 2.6 + Triton 3.2 on an A100 80GB PCIe (see its README). The Chapter 1 lab still contains an older hardware-tier recipe with smaller models for CPU-only machines.

## Case study and principles

[`case_study/`](case_study/README.md) re-measures the case study on Modal on all three GPUs in one command and checks the book's principles against the measurements. It separates the claims the book teaches (`principles.py`, each with the chapters that teach it and its sources) from the swappable settings (`experiment_config.py`: GPUs, models, sizes, package versions). The measured data every chapter quotes is in [`case_study/reference_results/`](case_study/reference_results/README.md).

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
├── chapter06/                    Triton kernels (pinned PyTorch 2.6 + Triton 3.2)
├── chapter07/                    distributed training (analytical cells + measured 2x A10G runs)
├── case_study/                   multi-GPU re-measurement, principles, reference results
├── tools/drift_check/            monthly check that the code and numbers still hold
└── ...                           one folder per chapter as the book is released
```

Each chapter folder has its own README and a Colab-ready notebook. Notebook section headings name the chapter section they accompany, so you can read with the notebook open beside you. Each README states the tested stack: Python 3.11, PyTorch 2.14.1 (CUDA 13.0), transformers 5.20.2 and sentence-transformers 6.2.0 for Chapters 1–5 and 7, and PyTorch 2.6 + Triton 3.2 for Chapter 6.

## Quick start

### Option 1: Google Colab

1. Open the notebook on GitHub and click **Open in Colab**.
2. *Runtime → Change runtime type → Hardware accelerator*: a T4 works for every chapter (Chapter 6 needs the pinned Triton on a T4; see its README); an A100 or H100 matches the book's reference numbers.
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
| 8  | Inference Serving and KV Cache Management          | _coming_              | continuous batching, PagedAttention, speculative decoding  |
| 9  | Model Compression: Quantization, Distillation, LoRA| _coming_              | post-training quantization end-to-end                      |
| 10 | Adaptation: PEFT, RLHF, Preference Optimization    | _coming_              | parameter-efficient fine-tuning under a budget             |
| 11 | The Accelerator Landscape and Portability          | _coming_              | NVIDIA / AMD / Intel / TPU / Trainium trade-offs           |
| 12 | Production Maturity, Observability, Stewardship    | _coming_              | the capstone, tying everything to the case study           |

Packt publishes the book in chapter installments through their Early Access Program. New folders show up here as each chapter clears editorial.

## Book editions

Each published edition of the book is a git tag (`edition-1`, `edition-2`, ...). The tag fixes the notebooks, the pinned versions and the reference results that edition quotes. To read the code exactly as printed:

```bash
git fetch --tags
git checkout edition-1
```

The default branch keeps moving with PyTorch, Triton and new GPUs. [`CHANGES_SINCE_PUBLICATION.md`](CHANGES_SINCE_PUBLICATION.md) records behaviour that changed after an edition went to print, and [`RELEASING.md`](RELEASING.md) describes how an edition is cut.

## What's *not* in this repository

Manuscript drafts, figure source files, and other working materials live in a local `chapters/` folder that is gitignored. Only finished, runnable code lands at the top level. If you fork the repo and find that folder missing, that is on purpose.

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
