# Drift check

A GitHub Actions workflow (`.github/workflows/drift-check.yml`) re-runs the case study on one Modal T4
on the 1st of every month and compares the result with the reference run the book quotes
(`case_study/reference_results/run2_2026-10-23/t4.json`). It catches the ways a book's code ages:
an API that was renamed, a default that changed, a library that broke an experiment, or numbers that
moved.

## What fails the check

`compare.py` fails the job when

- a principle in `case_study/principles.py` changes verdict (holds → FAILS or n/a, or the reverse);
- an experiment reports an error it did not report in the reference run (the known crash of
  `mode="reduce-overhead"` compiled decode is ignored; see `CHANGES_SINCE_PUBLICATION.md`);
- a measurement under `decode_batching`, `embedder.throughput` or `compiled_decode.*.decode_ms_per_token`
  moves by more than 20%, or disappears.

Numbers from the Chapter 2 lab's own decode loop depend heavily on the host machine (two runs of the same
stack on a T4 differed by up to 2x), so they appear as warnings and do not fail the job.

When the job fails, GitHub emails the repository owner. The report is in the job summary, and the full
results are attached to the run as an artifact for 90 days. Record anything real in
`CHANGES_SINCE_PUBLICATION.md`.

## One-time setup

1. Create a Modal token for CI: in the Modal dashboard, Settings → API Tokens → New Token (or run
   `modal token new` locally and read `~/.modal.toml`).
2. In GitHub: repository Settings → Secrets and variables → Actions → New repository secret. Add
   - `MODAL_TOKEN_ID`
   - `MODAL_TOKEN_SECRET`
3. The Modal workspace must already have the `huggingface` secret (`HF_TOKEN`) that the case study uses,
   with access to google/gemma-2b-it.
4. Make sure Actions notifications are on for your account (GitHub Settings → Notifications → Actions).

To run it now: Actions → Drift check → Run workflow. You can name a different reference folder there.

## Run the comparison locally

```bash
python tools/drift_check/compare.py case_study/results/t4.json \
    case_study/reference_results/run2_2026-10-23/t4.json --report drift_report.md
```

Cost: one T4 run of the case study (Chapter 2 lab plus all experiments) per month; check the Modal dashboard after the first run for the exact figure.
