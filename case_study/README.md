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
   - Chapter 4: eager decode (dynamic cache; static cache with transformers' automatic compilation
     off), the static cache with automatic compilation on, compiling the module and calling
     `.generate()`, and compiling `model.forward` with a static KV cache with CUDA graphs
     (`mode="reduce-overhead"`) and without (`mode="default"`). Each variant runs in its own child
     process, so a native crash in one compile mode does not lose the others.
   - Chapter 5: prefill and per-token decode against batch size, with achieved compute and bandwidth.
   - `experiments_extra.py` (see its docstring for the chapter map): first-call warm-up (Ch 2);
     Python loop vs vectorized sum, autograd overhead, DataLoader sweep (Ch 3); compiled encoder and
     prefill, GPT-2 `explain` output, cold/warm/regional compile start, eager vs compiled profiles
     (Ch 4); measured GEMM and copy-bandwidth roofs and kernel-launch overhead (Ch 5).
3. **Traces, first repeat only** (`results/traces/<gpu>/`):
   - `pytorch_profiler/*.json.gz` and `extra/gpt2_profile/*.json.gz`: Chrome traces. Open them at
     https://ui.perfetto.dev (drag the file in; .json.gz works as is). Each has a matching
     `key_averages()` table as `.txt`.
   - `chapter02_lab/*.json.gz`: the two traces the Chapter 2 lab writes itself.
   - `nsys/rag_decode.nsys-rep`: Nsight Systems report of prefill + 5 NVTX-labelled decode steps. Open
     it in the Nsight Systems GUI (free, macOS/Windows/Linux). The CSVs beside it are the kernel,
     CUDA API and NVTX summaries from `nsys stats`.
4. **Chapter 7, optional** (`--allreduce`): NCCL all-reduce bus bandwidth on one 8×A100 and one
   8×H100 container, written to `results/allreduce_<gpu>.json`.

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
modal run case_study/modal_case_study.py --repeats 5       # 5 independent containers per GPU
modal run case_study/modal_case_study.py --repeats 5 --allreduce   # plus the Chapter 7 8-GPU all-reduce job
```

Model weights are cached in the Modal volume `hpml-hf-cache`, so later runs skip the downloads.

## Output (`case_study/results/`)

- `<gpu>.json`: every measurement for one GPU, with the GPU, host CPU, driver and package versions.
- `chapter02_lab_executed_<gpu>.ipynb`: the executed Chapter 2 lab with all outputs.
- `principles_report.md` / `.json`: each principle per GPU (holds, FAILS, n/a) with its evidence.
- `master_tables.md` / `.json`: every case-study table the chapters quote, built by
  `python case_study/make_tables.py` from the `<gpu>.json` files.

With `--repeats N`, each repeat runs in its own container (and usually on a different host).
Every repeat is saved in `results/repeats/`; `<gpu>.json` becomes the median across repeats,
`<gpu>_spread.json` holds the min and max of every number, `master_tables.md` shows (min–max)
wherever repeats differ by more than 10%, and `principles_report.md` adds how many repeats each
principle held in. Overhead-bound numbers (batch-1 decode and encoding) vary most between hosts.

Each experiment keeps going if another fails and records the error in the JSON, so it can be
fixed and re-run on its own. For an edition release, copy the versions recorded under `env` into
`PIP_PACKAGES` so the published numbers can be reproduced.
