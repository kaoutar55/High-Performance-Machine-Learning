# Case study: principles and swappable experiments

This folder re-measures the book's RAG case study on several GPUs and checks that the book's
principles still hold on each one. It keeps two kinds of content apart:

| File | What it holds | How often it changes |
|---|---|---|
| `principles.py` | The claims the book teaches (decode is bandwidth-limited, batching amortizes weight reads, compile only helps compiled code, ...), each with the chapters that teach it and a check against measurements | Rarely; a principle that fails on new hardware is a finding for the book |
| `experiment_config.py` | GPUs, models, precision, workload sizes, package versions, datasheet peaks | Every refresh: new GPU, new model, new stack |
| `modal_case_study.py` | The runner: measures everything in the config, then evaluates every principle | Only when a new experiment is added |

To refresh the book for a new GPU or model, edit `experiment_config.py` and re-run. The numbers
change; `results/principles_report.md` shows whether the lessons still hold.

## What runs on each GPU

1. **The Chapter 2 lab, executed top to bottom** (`chapter02/chapter02_lab.ipynb`: Gemma-2B-IT in
   FP16, BGE-M3 embedder, FAISS): latency distribution, stage table, profiler table and Roofline
   placement.
2. **Extra experiments**:
   - Chapter 3: BGE-M3 throughput on 512-token chunks (FP32 vs FP16, batch 1–64), model shape,
     operator profile, and which attention kernel runs.
   - Chapter 4: eager decode, compiling the module and calling `.generate()`, and compiling
     `model.forward` with a static KV cache.
   - Chapter 5: prefill and per-token decode against batch size, with achieved compute and bandwidth.

## One-time setup

```bash
python3 -m venv ~/.venvs/hpml && source ~/.venvs/hpml/bin/activate
pip install --upgrade pip modal
modal setup                                   # log in to Modal in the browser
# Accept the Gemma license at huggingface.co/google/gemma-2b-it, create a read token, then:
modal secret create huggingface HF_TOKEN=<your_read_token>
```

## Run

From the repository root:

```bash
modal run case_study/modal_case_study.py                   # A100-80GB, H100 and T4 in parallel
modal run case_study/modal_case_study.py --gpus T4         # a subset, comma-separated
```

Model weights are cached in the Modal volume `hpml-hf-cache`, so later runs skip the downloads.

## Output (`case_study/results/`)

- `<gpu>.json`: every measurement for one GPU, with the GPU, driver and software versions.
- `chapter02_lab_executed_<gpu>.ipynb`: the executed Chapter 2 lab with all outputs.
- `principles_report.md` / `.json`: each principle per GPU (holds, FAILS, n/a) with its evidence.

Each experiment keeps going if another fails and records the error in the JSON, so it can be
fixed and re-run on its own. For an edition release, copy the versions recorded under `env` into
`PIP_PACKAGES` so the published numbers can be reproduced.
