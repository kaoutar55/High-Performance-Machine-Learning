# High-Performance Machine Learning

Companion code for *High-Performance Machine Learning* (Packt) by **Dr. Kaoutar El Maghraoui**.

A model that is accurate but too slow does not ship. Neither does one that is accurate but too expensive to serve, or one that crashes when traffic doubles. The book is built around that observation. It treats performance as a design concern from the first commit, and it walks through the techniques that actually move the needle: compilation, custom kernels, quantization, continuous batching, KV cache management, distributed training, accelerator portability.

This repo holds runnable notebooks for every chapter. The same multimodal RAG case study runs through all of them. You start with a 5–8 second naïve baseline on a free Colab T4 and finish under 1.5 seconds end-to-end on the same hardware. About 4.6× on Colab's slowest GPU, with no exotic infrastructure.

## Hardware tiers

The notebooks detect what you have and pick model sizes that fit. The pipeline is identical across tiers; only the model sizes change.

| Tier              | Exemplar             | Memory     | Embedding model        | Generation model         | Expected naïve TTFT |
|-------------------|----------------------|------------|------------------------|--------------------------|---------------------|
| Consumer CPU      | Any laptop           | ≥ 8 GB RAM | all-MiniLM-L6 (22 M)   | Qwen2.5-0.5B-Instruct    | ~10–30 s            |
| Consumer GPU 16 GB| Colab Free T4        | ~15 GB     | all-MiniLM-L6 (22 M)   | Qwen2.5-1.5B-Inst (4-bit)| ~3–8 s              |
| Datacenter 40 GB  | Colab Pro A100 / L40 | ~40 GB     | bge-base-en-v1.5       | Qwen2.5-3B-Instruct      | ~1–3 s              |
| Datacenter 80 GB+ | A100 80 / H100       | ~80 GB     | bge-large-en-v1.5      | Qwen2.5-7B-Instruct      | ~0.5–1 s            |

The model names above are 2026 picks. They will rotate as Hugging Face publishes new checkpoints. What does not rotate is the structural claim: same pipeline, scaled model size, same lessons.

## Repository layout

```
.
├── README.md                this file
├── LICENSE                  MIT
├── chapter02/
│   └── HPML_Ch02_Lab.ipynb  companion notebook for Chapter 2
└── ...                      one folder per chapter as the book is released
```

Each chapter folder ships a single Colab-ready `.ipynb`. Section numbers in the notebook match the section numbers in the chapter, so you can read with the notebook open beside you. Pinned versions live inline at the top of every notebook, which means a cold runtime reproduces what's in the book without you maintaining a separate lockfile.

## Quick start

### Option 1: Google Colab

1. Open the notebook on GitHub and click **Open in Colab**.
2. *Runtime → Change runtime type → Hardware accelerator: T4 GPU*.
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

Each notebook installs its own pinned stack the first time you run it. If you have a CUDA GPU, it gets used. If not, the notebook falls back to the CPU recipe and the timings rescale accordingly.

## Chapter map

| #  | Title                                              | Notebook              | What you produce                                           |
|----|----------------------------------------------------|-----------------------|------------------------------------------------------------|
| 1  | The Imperative of Performance in Modern AI         | _conceptual_          | the four factors, hardware tiering, naïve RAG baseline     |
| 2  | A Methodology for Performance Engineering          | `chapter02/`          | profiler runs, Roofline placement, regression test         |
| 3  | PyTorch Execution Model and `torch.compile`        | _coming with the book_| graph capture, fused kernels, eager-vs-compiled gap        |
| 4  | Mixed-Precision and Quantization Foundations       | _coming_              | FP16/BF16/INT8/INT4 vs accuracy and throughput             |
| 5  | GPU Architecture and the Roofline Model            | _coming_              | hardware-aware reasoning, memory hierarchy                 |
| 6  | Custom Kernels: Triton, FlashAttention, FAISS GPU  | _coming_              | attention kernel walk-through, GPU vector search           |
| 7  | Distributed Training: DP, TP, PP, FSDP             | _coming_              | scaling without drowning in communication                  |
| 8  | Inference Serving and KV Cache Management          | _coming_              | continuous batching, PagedAttention, speculative decoding  |
| 9  | Model Compression: Quantization, Distillation, LoRA| _coming_              | post-training quantization end-to-end                      |
| 10 | Adaptation: PEFT, RLHF, Preference Optimization    | _coming_              | parameter-efficient fine-tuning under a budget             |
| 11 | The Accelerator Landscape and Portability          | _coming_              | NVIDIA / AMD / Intel / TPU / Trainium trade-offs           |
| 12 | Production Maturity, Observability, Stewardship    | _coming_              | the capstone, tying everything to the case study           |

Packt publishes the book in chapter installments through their Early Access Program. New folders show up here as each chapter clears editorial.

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
