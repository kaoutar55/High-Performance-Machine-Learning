# Reference results for the book's case study

Two runs of `modal run case_study/modal_case_study.py` on 23 October 2026, same code paths and the
same stack (PyTorch 2.14.1+cu130, CUDA 13.0, transformers 5.20.2, sentence-transformers 6.2.0,
NVIDIA driver 610.57.04), on Modal A100-SXM4-80GB, H100 80GB HBM3 and T4.

- `run2_2026-10-23/` is the **reference run** the chapters quote. It has the split Chapter 4
  decode variants and 10 timing iterations.
- `run1_2026-10-23/` is kept to show run-to-run variation. Its Chapter 4 "eager static cache" row
  was compiled by transformers' automatic compilation, so do not quote it as eager.

Each folder holds `<gpu>.json`, the executed Chapter 2 notebooks, `principles_report.md` (regenerated
with the current `principles.py`) and `master_tables.md`.

## Run-to-run variation

Measurements limited by host-side overhead moved by up to 2x between the two runs; measurements
limited by GPU compute or memory bandwidth moved by about 5%:

| Measurement | GPU | Run 1 | Run 2 |
|---|---|---|---|
| Decode step, batch 1 (ms), overhead-bound | A100 | 8.98 | 17.46 |
| Embedder FP16 batch 1 (chunks/s), overhead-bound | A100 | 135.9 | 66.9 |
| Prefill, batch 1 (TFLOP/s), compute-bound | A100 | 126.4 | 119.4 |
| Decode step, batch 1 (ms), near the bandwidth floor | T4 | 24.38 | 24.63 |

Modal's sandbox does not expose the host CPU model (`env.cpu` reads "unknown"), so the host cannot
be compared directly. Quote ratios and principle verdicts with confidence; quote absolute
overhead-bound numbers with the run they came from.
