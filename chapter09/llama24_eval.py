"""Chapter 9: the Table 9.11 evaluation for the Llama-3.1-8B-Instruct 2:4 SparseGPT re-run, on one GPU outside Modal.

reference() and quality() follow gemma24_eval.py with the base model as a parameter; make_llm(), free() and the
modal_quant.py helpers (kl_rows, topk_from, score_ids, answer_scores, paired, _norm, gold_answers) are reused from it.
lm-eval runs the two commands of the published run (results/quality_llama.json, variant bf16), and the per-item
parsing below is modal_quant.lm_eval()'s, copied verbatim.

    python llama24_eval.py reference --base meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --prompts P --out REF
    python llama24_eval.py quality --src SRC --label L --dtype bfloat16 --ref REF --prompts P --out q.json
    python llama24_eval.py diffs --baseline q_dense.json --variant q_sparse.json --out d.json
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gemma24_eval import free, make_llm, mq  # noqa: E402
from gemma24_rerun import env_block, eval_prompts  # noqa: E402

# per-item mean -> the aggregate metric it must equal (the Table 9.11 metrics)
CLAIMED = {"gsm8k_cot_llama": "exact_match,flexible-extract", "ifeval": "prompt_level_strict_acc,none",
           "arc_challenge_llama": "exact_match,strict_match", "arc_challenge": "acc,none"}
TABLE_TASKS = {"gsm8k_cot_llama": "GSM8K", "ifeval": "IFEval", "arc_challenge_llama": "ARC-C"}


def reference(base, prompts, dtype, out_dir):
    """gemma24_eval.reference() with the base model as a parameter."""
    import numpy as np
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    f = mq()
    os.makedirs(out_dir, exist_ok=True)
    llm = make_llm(base, dtype)
    tok = llm.get_tokenizer()
    p_ids = [tok(p, add_special_tokens=True)["input_ids"] for p in eval_prompts(prompts)]
    gen = llm.generate([TokensPrompt(prompt_token_ids=p) for p in p_ids], SamplingParams(temperature=0.0, max_tokens=128))
    r_ids = [list(o.outputs[0].token_ids) for o in gen]
    scored = llm.generate([TokensPrompt(prompt_token_ids=p + r) for p, r in zip(p_ids, r_ids)],
                          SamplingParams(max_tokens=1, prompt_logprobs=20))
    ids, lps = [], []
    for p, r, o in zip(p_ids, r_ids, scored):
        rows = [f["topk_from"](o.prompt_logprobs[len(p) + t]) for t in range(len(r))]
        ids.append(np.array([x[0] for x in rows], dtype=np.int32))
        lps.append(np.array([x[1] for x in rows]))
    noise, method = f["score_ids"](llm, p_ids, r_ids, ids)
    np.savez(os.path.join(out_dir, "reference.npz"), p_ids=np.array(p_ids, dtype=object),
             r_ids=np.array(r_ids, dtype=object), ids=np.array(ids, dtype=object), lps=np.array(lps, dtype=object),
             texts=np.array([o.outputs[0].text for o in gen], dtype=object))
    nf = f["kl_rows"](lps, noise)
    nf["method"] = method
    info = {"base": base, "dtype": dtype, "prompts": len(p_ids), "answer_tokens": int(sum(len(r) for r in r_ids)),
            "noise_floor": nf, "env": env_block()}
    json.dump(info, open(os.path.join(out_dir, "reference.json"), "w"), indent=2)
    free(llm)
    print(json.dumps({"reference": base, "answer_tokens": info["answer_tokens"], "noise_kl": nf["kl_mean"],
                      "noise_top1": nf["top1_agreement"]}), flush=True)


def lm_eval(src, dtype, out_dir):
    """The two lm-eval commands of the published run, then modal_quant.lm_eval()'s parsing (verbatim)."""
    args = f"pretrained={src},dtype={dtype},max_model_len=4096,gpu_memory_utilization=0.85,seed=0"
    runs = {}
    for tag, tasks, flags in (("gen", "gsm8k_cot_llama,ifeval,arc_challenge_llama",
                               ["--apply_chat_template", "--fewshot_as_multiturn"]),
                              ("ll", "arc_challenge,wikitext", [])):
        a = args + (",add_bos_token=True" if tag == "ll" else "")
        cmd = ["lm_eval", "--model", "vllm", "--model_args", a, "--tasks", tasks, "--batch_size", "auto",
               "--seed", "1234", "--output_path", f"{out_dir}/{tag}", "--log_samples"] + flags
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        runs[tag] = {"cmd": " ".join(cmd), "returncode": r.returncode, "wall_s": round(time.time() - t0, 1),
                     "stderr_tail": r.stderr[-3000:] if r.returncode else ""}
    # ---- verbatim from modal_quant.lm_eval()
    metrics, per_item = {}, {}
    for f in pathlib.Path(out_dir).rglob("results_*.json"):
        for task, m in json.load(open(f)).get("results", {}).items():
            metrics[task] = {k: v for k, v in m.items() if isinstance(v, (int, float))}
    keymetric = {"gsm8k_cot_llama": "exact_match", "ifeval": "prompt_level_strict_acc",
                 "arc_challenge_llama": "exact_match", "arc_challenge": "acc"}
    for f in pathlib.Path(out_dir).rglob("samples_*.jsonl"):
        task = next((t for t in keymetric if f.name.startswith(f"samples_{t}_")), None)
        if not task:
            continue
        vals = {}
        for line in open(f):
            s = json.loads(line)
            v = s.get(keymetric[task])
            if v is None:                                  # e.g. per-filter names such as "exact_match,strict-match"
                v = next((s[k] for k in s if k.startswith(keymetric[task])), None)
            if isinstance(v, list):
                v = v[0]
            if v is not None:
                vals[int(s["doc_id"])] = float(v)
        per_item[task] = vals
    # ---- end of the verbatim part
    checks = {}
    for task, key in CLAIMED.items():
        items = per_item.get(task, {})
        agg = metrics.get(task, {}).get(key)
        mean = sum(items.values()) / len(items) if items else None
        checks[task] = {"claimed_metric": key, "aggregate": agg, "per_item_mean": mean, "n": len(items),
                        "equal": mean is not None and agg is not None and abs(mean - agg) < 1e-9}
    return {"runs": runs, "metrics": metrics, "per_item": per_item, "per_item_checks": checks}


def quality(src, dtype, ref_dir, prompts, out, label):
    """gemma24_eval.quality() with the Table 9.11 lm-eval protocol."""
    import numpy as np
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    f = mq()
    t0 = time.time()
    res = {"label": label, "src": os.path.basename(src.rstrip("/")), "dtype": dtype, "env": env_block()}
    res["lm_eval"] = lm_eval(src, dtype, f"/tmp/{os.environ.get('USER', 'u')}/lmeval_{label}_{os.getpid()}")
    ref = np.load(os.path.join(ref_dir, "reference.npz"), allow_pickle=True)
    p_ids, r_ids, ids, lps = list(ref["p_ids"]), list(ref["r_ids"]), list(ref["ids"]), list(ref["lps"])
    llm = make_llm(src, dtype)
    gen = llm.generate([TokensPrompt(prompt_token_ids=list(p)) for p in p_ids],
                       SamplingParams(temperature=0.0, max_tokens=128))
    same, prefix = [], []
    for o, r in zip(gen, r_ids):
        g, r = list(o.outputs[0].token_ids), list(r)
        same.append(g == r)
        n = 0
        while n < min(len(g), len(r)) and g[n] == r[n]:
            n += 1
        prefix.append(n)
    res["generation"] = {"identical_answers": float(np.mean(same)), "mean_common_prefix_tokens": float(np.mean(prefix)),
                         "answers": len(same)}
    golds = f["gold_answers"](eval_prompts(prompts))
    b_c, b_f = f["answer_scores"](list(ref["texts"]), golds)
    v_c, v_f = f["answer_scores"]([o.outputs[0].text for o in gen], golds)
    res["squad"] = {"prompts_with_reference": int(sum(1 for g in golds if g)),
                    "contains_answer": f["paired"](b_c, v_c), "token_f1": f["paired"](b_f, v_f),
                    "per_item": {"baseline_contains": b_c, "variant_contains": v_c}}
    q_lps, method = f["score_ids"](llm, p_ids, r_ids, ids)
    res["kl"] = f["kl_rows"](lps, q_lps)
    res["kl"]["method"] = method
    res["kl"]["reference"] = json.load(open(os.path.join(ref_dir, "reference.json")))["dtype"]
    free(llm)
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(out, "w"), indent=2)
    m, c = res["lm_eval"]["metrics"], res["lm_eval"]["per_item_checks"]
    print(json.dumps({"label": label, "gsm8k_flex": m.get("gsm8k_cot_llama", {}).get(CLAIMED["gsm8k_cot_llama"]),
                      "ifeval_strict": m.get("ifeval", {}).get(CLAIMED["ifeval"]),
                      "arc_llama": m.get("arc_challenge_llama", {}).get(CLAIMED["arc_challenge_llama"]),
                      "wikitext_ppl": m.get("wikitext", {}).get("word_perplexity,none"), "kl": res["kl"]["kl_mean"],
                      "top1": res["kl"]["top1_agreement"], "squad": res["squad"]["contains_answer"]["variant"],
                      "per_item_checks_equal": all(x["equal"] for x in c.values())}), flush=True)
    bad = [t for t, x in c.items() if not x["equal"]]
    if bad:
        raise SystemExit(f"per-item means do not equal the claimed aggregates for {bad} (results saved)")


def diffs(baseline, variant, out):
    """Paired differences in points (x 100) with SE and flips, over doc_ids present in both, as in Table 9.11."""
    f = mq()
    b, v = json.load(open(baseline)), json.load(open(variant))
    res = {"baseline": b["label"], "variant": v["label"], "tasks": {}}
    for task, name in list(TABLE_TASKS.items()) + [("arc_challenge", "ARC-C (acc, log-likelihood)")]:
        bi, vi = b["lm_eval"]["per_item"].get(task, {}), v["lm_eval"]["per_item"].get(task, {})
        common = sorted(set(bi) & set(vi), key=int)
        p = f["paired"]([bi[k] for k in common], [vi[k] for k in common])
        res["tasks"][name] = {**p, "diff_points": 100 * p["diff"],
                              "diff_se_points": 100 * p["diff_se"] if p["diff_se"] is not None else None}
    sq = v["squad"]["contains_answer"]
    res["tasks"]["SQuAD contains"] = {**sq, "diff_points": 100 * sq["diff"],
                                      "diff_se_points": 100 * sq["diff_se"] if sq["diff_se"] is not None else None}
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps({k: (round(x["diff_points"], 1), round(x["diff_se_points"] or 0, 1)) for k, x in res["tasks"].items()}),
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["reference", "quality", "diffs"])
    ap.add_argument("--base", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--src")
    ap.add_argument("--label")
    ap.add_argument("--dtype", choices=["bfloat16"], default="bfloat16")
    ap.add_argument("--ref")
    ap.add_argument("--prompts")
    ap.add_argument("--baseline")
    ap.add_argument("--variant")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.what == "reference":
        reference(a.base, a.prompts, a.dtype, a.out)
    elif a.what == "quality":
        quality(a.src, a.dtype, a.ref, a.prompts, a.out, a.label)
    else:
        diffs(a.baseline, a.variant, a.out)
