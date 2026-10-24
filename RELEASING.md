# Releasing a book edition

Each edition of the book is a git tag that fixes the notebooks, the pinned stack and the measured
numbers the printed text quotes. Readers check out the tag to get exactly what the book describes.

## 1. Pin the stack

1. Pick the reference run the edition quotes, for example
   `case_study/reference_results/<run>/a100_80gb.json`.
2. Copy the package versions from `env.packages` in that file into `PIP_PACKAGES` in
   `case_study/experiment_config.py` (exact `==` pins for torch, transformers, sentence-transformers,
   accelerate, datasets, faiss-cpu, numpy and the notebook tools).
3. State the same versions in every chapter README and notebook setup cell. Chapter 6 keeps its own pin
   (PyTorch 2.6 + Triton 3.2).

## 2. Regenerate the reference results

```bash
rm -rf case_study/results
modal run case_study/modal_case_study.py --repeats 5 --allreduce
```

Check `case_study/results/principles_report.md`: every principle should hold in 5 of 5 repeats on every
GPU, or the edition's text must say where it does not. Copy `case_study/results/` (without the large
`traces/` folder, unless the edition needs them) to `case_study/reference_results/<edition-date>/` and
add a line to `case_study/reference_results/README.md` naming it as the edition's reference run.
Rebuild the manuscript tables from that folder.

## 3. Tag and release

```bash
git checkout main && git pull
git tag -a edition-1 -m "High-Performance Machine Learning, first edition (Packt)"
git push origin edition-1
gh release create edition-1 --title "First edition" \
  --notes "Code, pinned versions and measured results as printed in the first edition."
```

## 4. Tell Packt

Send the tag URL, for example
`https://github.com/kaoutar55/High-Performance-Machine-Learning/tree/edition-1`, for the book's
"Code in Action" and "Download the example code" pages. The default branch keeps moving; the tag does not.

## 5. After release

Start `CHANGES_SINCE_PUBLICATION.md` entries from the drift check and reader issues. A new edition
repeats steps 1–4 with the next tag (`edition-2`).
