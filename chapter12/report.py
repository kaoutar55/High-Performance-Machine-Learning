"""Collect measured results into results/case_study.json, the file the chapter build reads.

Schema (every key optional; missing values print as '–' in the chapter):
{
 "E0".."E7": {"accuracy", "task_p50_s", "task_p95_s", "steps", "hit_rate", "visual_tokens", "goodput", "gpus",
              "gpu_seconds_per_task"},
 "R0": {"text_ndcg10", "page_ndcg10", "hybrid_ndcg10", "page4bit_ndcg10"},
 "E8": {"rr_hit_rate", "affinity_hit_rate", "prefix_hit_rate", "rr_p95_s", "affinity_p95_s", "prefix_p95_s"},
 "E9": {"gpus": 1, "gpu_seconds_per_task": {"L40S", "A100", "H100"}, "goodput_tasks_per_s", "gpu_models", ...}
 "R1": {"baseline_cv_pct", "upgrade_verdict"},   # verdict: "PASS" or the failing metrics, as a phrase
 "R2": {"goodput_change_pct"}
}
Goodput = highest tested rate at which >= 90% of tasks met the task objective in all three repeats; the row values
are the medians of the three repeats at that rate.

GPUs: E0-E6 use two H100s (vLLM on one, the retriever on a second H100 of the same node); E7 colocates the retriever
with vLLM on one GPU (retriever.colocate in configs/experiments.yaml). E9 runs the E7 configuration on one GPU.
GPU-seconds per task = GPUs / goodput, at utilization 1.0. No prices are stored: the cost per task at a price P per
GPU-hour is gpu_seconds_per_task x P / 3600 (cost.py computes the same with a utilization factor).

    python report.py && cp results/case_study.json ../../results/case_study.json   # then rebuild the chapter
"""
import glob
import json
import os
import statistics
from collections import defaultdict

import yaml

from evaluate import score
from loadgen_agent import summarize

cfg = yaml.safe_load(open("configs/experiments.yaml"))
obj = cfg["objectives"]["task_p95_single_s"]
queries = [json.loads(l) for l in open("data/queries.jsonl")]


def n_gpus(exp):
    """GPUs used by a ladder experiment: one when the retriever shares the vLLM GPU, otherwise two."""
    colocate = cfg.get(exp, {}).get("retriever", {}).get("colocate", cfg["common"]["retriever"]["colocate"])
    return 1 if colocate else 2


out = {}
runs = defaultdict(lambda: defaultdict(list))            # exp -> rate -> [rows per repeat]
for p in sorted(glob.glob("results/E[0-7]_H100_single_*_*.jsonl")):   # sorted: list index = repeat number
    if p.endswith("_trace.jsonl"):
        continue
    exp, _, _, rate, rep = os.path.basename(p)[:-6].split("_")
    runs[exp][float(rate)].append([json.loads(l) for l in open(p)])
for exp, by_rate in runs.items():
    good = [r for r, reps in by_rate.items() if len(reps) == 3 and all(summarize(x, obj)["met_objective"] >= 0.9 for x in reps)]
    if not good:
        continue
    rate = max(good)
    s = [summarize(x, obj) for x in by_rate[rate]]
    def judged(rate_, i, rows_):                         # judged verdicts from evaluate.py, when present
        p = f"results/{exp}_H100_single_{rate_}_{i}.judged.json"
        return json.load(open(p))["accuracy"] if os.path.exists(p) else score(rows_, queries)[0]
    acc = [judged(rate, i, x) for i, x in enumerate(by_rate[rate])]
    vis = [statistics.mean(r.get("images", 0) for r in x if r.get("ok")) for x in by_rate[rate]]
    out[exp] = {"accuracy": statistics.median(acc), "task_p50_s": statistics.median(v["task_p50_s"] for v in s),
                "task_p95_s": statistics.median(v["task_p95_s"] for v in s), "steps": statistics.median(v["steps"] for v in s),
                "hit_rate": statistics.median(v["hit_rate"] for v in s), "goodput": rate,
                "visual_tokens": None, "images_per_task": statistics.median(vis),
                "gpus": n_gpus(exp), "gpu_seconds_per_task": n_gpus(exp) / rate}
for extra in ("R0", "E8", "E9", "R1", "R2"):            # written by the respective scripts, merged here
    p = f"results/{extra}.json"
    if os.path.exists(p):
        out[extra] = json.load(open(p))
if "E9" in out:                                          # keep goodput and GPU-seconds; drop the dollar values
    e9 = out["E9"]
    good = e9.get("goodput_tasks_per_s", {})
    out["E9"] = {"gpus": 1, "gpu_seconds_per_task": {g: 1 / r for g, r in good.items()}, "goodput_tasks_per_s": good,
                 "gpu_models": e9.get("gpus"),
                 **{k: e9[k] for k in ("share_met_by_rate", "p50_p95_s_at_goodput") if k in e9},
                 "note": "E7 configuration on one GPU (FP8 generator, 4-bit page retriever on the same GPU). L40S: "
                         "gpu-memory-utilization 0.70 (0.80 does not fit next to the retriever). L40S and A100: 1 "
                         "repeat per rate (0.5, 1, 2 tasks/s); H100: 3 repeats (from E7). GPU-seconds per task at "
                         "goodput, utilization 1.0."}
json.dump(out, open("results/case_study.json", "w"), indent=1)
print(json.dumps(out, indent=1))
