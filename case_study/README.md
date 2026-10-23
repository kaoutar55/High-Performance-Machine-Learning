# Case-study re-measurement on Modal

This folder re-measures the book's RAG case study on a cloud GPU so every chapter can quote one
consistent set of numbers.

`modal_case_study.py` runs two jobs in parallel on the GPU you choose:

1. **The Chapter 2 lab, executed top to bottom** (`chapter02/chapter02_lab.ipynb`: Gemma-2B-IT in
   FP16, BGE-M3 embedder, FAISS). This produces the latency distribution (Table 2.6), the stage
   table and waterfall (Table 2.7, Figure 2.9), the profiler table (Table 2.4) and the Roofline
   placement (Table 2.8).
2. **Extra measurements** that no notebook produces yet:
   - Chapter 3: BGE-M3 throughput on 512-token chunks (FP32 vs FP16, batch 1–64), model shape,
     operator profile, and which attention backend runs on the GPU.
   - Chapter 4: Gemma decode in eager mode and with `torch.compile`, comparing compiling the module
     and calling `.generate()` with compiling `model.forward` with a static KV cache.
   - Chapter 5: Gemma prefill and per-token decode time against batch size, with achieved bandwidth.

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
source ~/.venvs/hpml/bin/activate
modal run case_study/modal_case_study.py                    # NVIDIA T4 (Colab free-tier reference)
modal run case_study/modal_case_study.py --gpu A100-80GB    # same measurements on an A100
```

Model weights are cached in the Modal volume `hpml-hf-cache`, so later runs skip the downloads.

## Output

`case_study/results/` receives:

- `case_study_results_<gpu>.json`: all measurements plus the software and GPU versions used.
- `chapter02_lab_executed_<gpu>.ipynb`: the executed Chapter 2 lab with every output.

Each job keeps going if one section fails, and records the error in the JSON so it can be fixed
and re-run on its own.
