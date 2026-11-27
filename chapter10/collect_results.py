"""Collect the Chapter 10 measurements into results/ch10_results.json, the file the chapter build reads.

Schema (keys match content_e.py and figs_ch10.py):
  train[ "<Model>|<Method>|<GPU>" ]   = {peak_mem_gb, tokens_per_s, gpu_hours}
  quality[ "<Model>|<Variant>" ]      = {hotpot_em: [diff, se], abstain_f1: [diff, se], gsm8k, ifeval, mmlu,
                                         forget_avg: [diff, se], kl}
  rank[ "<r>" ]                       = {em, forget, em_rs, forget_rs}   (each [diff, se])
  serve[ "<Configuration>" ]          = {tpot_ms, capacity_rps, ttft_p90_ms}
  distill[ "<Model row>" ]            = {hotpot_em, ifeval, tpot_ms, gpu_hours}
  rebase[ "<Generator>" ]             = {em_base, em_adapter, gpu_hours, tpot_ms}
  probe[ "<Variant>" ]                = {acc}
  spectrum                            = {label: [singular values]}
  verdict[ key ]                      = {measured, verdict}   (filled in by the author after reading the results)
"""
import glob
import json
import math
import os

from common import RESULTS, RUNS

NAMES = {"gemma-2b": "Gemma-2B", "llama-3.1-8b": "Llama-3.1-8B"}
METH = {"lora": "LoRA r=16", "qlora": "QLoRA r=16", "full": "Full FT"}
GPU = {"H100": "H100", "A100-80GB": "A100", "T4": "T4"}


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def paired(a, b):
    d = [float(x) - float(y) for x, y in zip(a, b)]
    n = len(d); m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / max(n - 1, 1))
    return [100 * m, 100 * sd / math.sqrt(n)]


def kl_top20(run_file, base_file):
    """Per-token KL(base || run) on the base model's top-20 support, averaged over all prompt positions."""
    r, b = jl(run_file), jl(base_file)
    if not (r and b):
        return None
    tot, n = 0.0, 0
    for pr, pb in zip(r["prompt_logprobs"], b["prompt_logprobs"]):
        for dr, db in zip(pr, pb):
            if not db or not dr:
                continue
            floor = min(dr.values())
            for tok, lb in db.items():
                tot += math.exp(lb) * (lb - dr.get(tok, floor))
            n += 1
    return tot / max(n, 1)


def lmeval_samples(tag, task):
    fs = glob.glob(f"{RESULTS}/lmeval/{tag}/{task}/**/samples_*.jsonl", recursive=True)
    if not fs:
        return None
    key = {"gsm8k_cot": "exact_match", "ifeval": "prompt_level_strict_acc", "mmlu": "acc", "arc_challenge": "acc"}[task]
    # A group task (MMLU: 57 subtasks) writes one samples file per subtask: use every file of the latest run,
    # ordered by subtask and doc_id so that a run and its base line up item by item.
    stamp = sorted(f.rsplit("_", 1)[-1] for f in fs)[-1]
    rows = [dict(json.loads(l), _f=os.path.basename(f).rsplit("_", 1)[0])
            for f in sorted(x for x in fs if x.endswith(stamp)) for l in open(f)]
    rows.sort(key=lambda r: (r["_f"], r["doc_id"]))
    return [float(r.get(key, r.get("acc", 0))) for r in rows]


def main():
    R = {"train": {}, "quality": {}, "serve": {}, "distill": {}, "rebase": {}, "rank": {}, "probe": {}}
    for m in glob.glob(f"{RUNS}/*/train_metrics.json"):
        t = json.load(open(m))
        if "model" not in t or t.get("seed", 0) != 0 or t.get("rank") != 16 or t.get("rslora"):
            continue
        if any(x in t.get("run", "") for x in ("-mix-", "-probe", "-lr", "-p2eval")):   # not the canonical training-table runs
            continue
        meth = METH.get(t["method"])
        if t["method"] == "full" and t["model"] == "llama-3.1-8b":
            meth = "Full FT (FSDP, 2 GPUs)"
        if meth and t["model"] in NAMES:
            R["train"][f"{NAMES[t['model']]}|{meth}|{GPU.get(t['gpu'], t['gpu'])}"] = {
                k: t[k] for k in ("peak_mem_gb", "tokens_per_s", "gpu_hours")}

    def q(model, run, label):
        base = jl(f"{RESULTS}/eval/{model}-base-hotpot.json"); cur = jl(f"{RESULTS}/eval/{run}-hotpot.json")
        if not (base and cur):
            return
        e = {"hotpot_em": paired(cur["em"], base["em"])}
        ab, abb = jl(f"{RESULTS}/eval/{run}-abstain.json"), jl(f"{RESULTS}/eval/{model}-base-abstain.json")
        if ab and abb:
            e["abstain_f1"] = [ab["f1"] - abb["f1"], float("nan")]
        fsum = []
        for task, k in (("gsm8k_cot", "gsm8k"), ("ifeval", "ifeval"), ("mmlu", "mmlu"), ("arc_challenge", "arc_c")):
            a, b = lmeval_samples(run, task), lmeval_samples(f"{model}-base", task)
            if a and b and len(a) == len(b):
                e[k] = paired(a, b); fsum.append(e[k])
        if fsum:
            e["forget_avg"] = [sum(x[0] for x in fsum) / len(fsum),
                               math.sqrt(sum(x[1] ** 2 for x in fsum)) / len(fsum)]
        kl = kl_top20(f"{RESULTS}/eval/{run}-kl.json", f"{RESULTS}/eval/{model}-base-kl.json")
        if kl is not None:
            e["kl"] = kl
        R["quality"][f"{NAMES[model]}|{label}"] = e

    for model in NAMES:
        q(model, f"{model}-lora-r16-s0-H100", "LoRA r=16")
        q(model, f"{model}-qlora-r16-s0-H100", "QLoRA r=16")
        q(model, f"{model}-full-r16-s0-H100", "Full FT")
    q("llama-3.1-8b", "llama-3.1-8b-lora-r16-mix-s0-H100", "LoRA r=16 + 10% general data")

    for f in glob.glob(f"{RESULTS}/serve/llama-3.1-8b-*-H100.json"):
        s = json.load(open(f))
        name = {"base": "Base model", "merged": "Merged adapter"}.get(s["mode"],
                                                                     f"Unmerged, {s['adapters']} adapter"
                                                                     + ("s" if s["adapters"] > 1 else ""))
        R["serve"][name] = {k: s[k] for k in ("tpot_ms", "capacity_rps", "ttft_p90_ms")}

    sp = jl(f"{RESULTS}/spectrum.json")
    if sp:
        R["spectrum"] = sp
    old = jl(f"{RESULTS}/ch10_results.json") or {}
    for k in ("distill", "rebase", "rank", "probe", "verdict"):   # filled by later steps or by hand
        if old.get(k):
            R[k] = old[k]
    json.dump(R, open(f"{RESULTS}/ch10_results.json", "w"), indent=1)
    print("wrote", f"{RESULTS}/ch10_results.json", {k: len(v) for k, v in R.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
