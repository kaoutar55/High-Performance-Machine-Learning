"""Chapter 9: the published quality evaluation (modal_quant.py reference() and quality()) for the Gemma-2B 2:4 re-run,
on one GPU outside Modal, with the dtype as a parameter.

kl_rows, topk_from, score_ids, answer_scores, paired, _norm and gold_answers are read from modal_quant.py's source and
executed here (modal_quant.py imports modal). lm-eval runs the published log-likelihood tasks (arc_challenge,
wikitext) with the published arguments; IFEval is not part of this re-run.

    python gemma24_eval.py reference --prompts prompts_single.jsonl --dtype bfloat16 --out ref_dir
    python gemma24_eval.py quality --src MODEL_OR_DIR --dtype bfloat16 --ref ref_dir --prompts ... --out q.json
"""
import argparse
import ast
import gc
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gemma24_rerun import env_block, eval_prompts  # noqa: E402

WANT = {"kl_rows", "topk_from", "score_ids", "answer_scores", "paired", "_norm", "gold_answers"}


def mq():
    src = (HERE / "modal_quant.py").read_text()
    lines = src.splitlines(keepends=True)
    parts = ["K = 20\n"]
    for n in ast.parse(src).body:
        if isinstance(n, ast.FunctionDef) and n.name in WANT:
            parts.append("".join(lines[n.lineno - 1:n.end_lineno]))
    ns = {}
    exec(compile("\n\n".join(parts), str(HERE / "modal_quant.py"), "exec"), ns)
    assert WANT <= set(ns), WANT - set(ns)
    return ns


def make_llm(src, dtype):
    from vllm import LLM
    return LLM(model=src, dtype=dtype, max_model_len=2048, max_logprobs=20, enable_prefix_caching=False,
               gpu_memory_utilization=0.85, seed=0)


def free(llm):
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:  # noqa: BLE001
        pass
    del llm
    gc.collect()
    import torch
    torch.cuda.empty_cache()


def reference(prompts, dtype, out_dir):
    """modal_quant.reference() with the dtype as a parameter."""
    import numpy as np
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    f = mq()
    os.makedirs(out_dir, exist_ok=True)
    llm = make_llm("google/gemma-2b-it", dtype)
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
    info = {"dtype": dtype, "prompts": len(p_ids), "answer_tokens": int(sum(len(r) for r in r_ids)),
            "noise_floor": nf, "env": env_block()}
    json.dump(info, open(os.path.join(out_dir, "reference.json"), "w"), indent=2)
    free(llm)
    print(json.dumps({"reference": dtype, "answer_tokens": info["answer_tokens"], "noise_kl": nf["kl_mean"]}), flush=True)


def lm_eval_ll(src, dtype, out_dir):
    args = (f"pretrained={src},dtype={dtype},max_model_len=4096,gpu_memory_utilization=0.85,seed=0,"
            "add_bos_token=True")
    cmd = ["lm_eval", "--model", "vllm", "--model_args", args, "--tasks", "arc_challenge,wikitext", "--batch_size",
           "auto", "--seed", "1234", "--output_path", out_dir, "--log_samples"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    res = {"cmd": " ".join(cmd), "returncode": r.returncode, "wall_s": round(time.time() - t0, 1),
           "stderr_tail": r.stderr[-3000:] if r.returncode else "", "metrics": {}}
    for f in pathlib.Path(out_dir).rglob("results_*.json"):
        for task, m in json.load(open(f)).get("results", {}).items():
            res["metrics"][task] = {k: v for k, v in m.items() if isinstance(v, (int, float))}
    return res


def quality(src, dtype, ref_dir, prompts, out, label, skip_lm_eval=False):
    """modal_quant.quality() with the dtype as a parameter (no IFEval)."""
    import numpy as np
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    f = mq()
    t0 = time.time()
    res = {"label": label, "src": os.path.basename(src.rstrip("/")), "dtype": dtype, "env": env_block()}
    if not skip_lm_eval:
        res["lm_eval"] = lm_eval_ll(src, dtype, f"/tmp/{os.environ.get('USER', 'u')}/lmeval_{label}_{os.getpid()}")
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
                    "contains_answer": f["paired"](b_c, v_c), "token_f1": f["paired"](b_f, v_f)}
    q_lps, method = f["score_ids"](llm, p_ids, r_ids, ids)
    res["kl"] = f["kl_rows"](lps, q_lps)
    res["kl"]["method"] = method
    res["kl"]["reference"] = json.load(open(os.path.join(ref_dir, "reference.json")))["dtype"]
    res["sample_answers"] = [o.outputs[0].text[:200] for o in gen[:3]]
    free(llm)
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(out, "w"), indent=2)
    m = res.get("lm_eval", {}).get("metrics", {})
    print(json.dumps({"label": label, "dtype": dtype, "word_ppl": m.get("wikitext", {}).get("word_perplexity,none"),
                      "arc_acc": m.get("arc_challenge", {}).get("acc,none"), "kl": res["kl"]["kl_mean"],
                      "top1": res["kl"]["top1_agreement"],
                      "squad_contains": res["squad"]["contains_answer"]["variant"]}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["reference", "quality"])
    ap.add_argument("--src")
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], required=True)
    ap.add_argument("--ref")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--label")
    ap.add_argument("--skip-lm-eval", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.what == "reference":
        reference(a.prompts, a.dtype, a.out)
    else:
        quality(a.src, a.dtype, a.ref, a.prompts, a.out, a.label, a.skip_lm_eval)
