"""Chapter 9: re-run of the Gemma-2B-IT 2:4 pruning results on one GPU, outside Modal.

Question: are the published 2:4 perplexities (magnitude 1,275, SparseGPT 6.6e5, Wanda 1.7e11, against 28.5 for FP16)
a numerical failure? The pruned models were built in BF16 and evaluated in FP16.

The pruning code is quantize.py's own: calibration_set, recipe_for, magnitude_24 and sparsity_report are read from
its source and executed here (quantize.py itself imports modal), with the prompt-file path as the only change.

    python gemma24_rerun.py build --variant sparse24_wanda --prompts prompts_single.jsonl --out DIR
    python gemma24_rerun.py check --ckpt DIR --out check.json           # 2:4 mask, dense embedding, tensor hashes
    python gemma24_rerun.py ppl   --ckpt DIR --dtype bfloat16 --out p.json   # lm-eval wikitext, Transformers backend
    python gemma24_rerun.py acts  --ckpt DIR --prompts prompts_single.jsonl --out a.json   # per-layer activations
"""
import argparse
import ast
import datetime
import glob
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
MODEL = "google/gemma-2b-it"
FP16_MAX = 65504.0
EVAL_ROWS = (800, 1000)


# ------------------------------------------------------------------------------------------------ helpers
def load_quantize_funcs(prompts_path):
    """Execute the named functions of quantize.py without importing it (it imports modal)."""
    src = (HERE / "quantize.py").read_text()
    tree = ast.parse(src)
    want = {"calibration_set", "recipe_for", "magnitude_24", "sparsity_report", "oneshot_dense_embedding"}
    lines = src.splitlines(keepends=True)
    parts = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want:
            parts.append("".join(lines[node.lineno - 1:node.end_lineno]))
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) in ("N_CALIB", "DECODER_PROJ") for t in
                                                (node.targets[0].elts if isinstance(node.targets[0], ast.Tuple)
                                                 else node.targets)):
            parts.insert(0, "".join(lines[node.lineno - 1:node.end_lineno]))
    code = "\n\n".join(parts)
    assert code.count('"/data/prompts_single.jsonl"') == 1, "quantize.py prompt path changed"
    code = code.replace('"/data/prompts_single.jsonl"', repr(str(prompts_path)))
    ns = {"json": json, "os": os}
    exec(compile(code, str(HERE / "quantize.py"), "exec"), ns)
    missing = want - set(ns)
    if missing:
        raise SystemExit(f"quantize.py functions not found: {missing}")
    return ns


def versions():
    import importlib.metadata as md
    out = {}
    for k in ("torch", "transformers", "vllm", "llmcompressor", "compressed-tensors", "lm_eval", "datasets",
              "safetensors"):
        try:
            out[k] = md.version(k)
        except md.PackageNotFoundError:
            pass
    return out


def env_block():
    q = subprocess.getoutput("nvidia-smi --query-gpu=name,driver_version,power.max_limit --format=csv,noheader")
    cuda = subprocess.getoutput("nvidia-smi | grep -oE 'CUDA (UMD )?Version: [0-9.]+'").split(":")[-1].strip()
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "unknown")
    try:
        import torch
        tcuda = torch.version.cuda
    except Exception:  # noqa: BLE001
        tcuda = None
    return {"date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "gpu": q.split(",")[0].strip(), "driver": q.split(",")[1].strip() if "," in q else q,
            "gpu_power_limit": q.split(",")[-1].strip(), "driver_cuda_version": cuda, "torch_cuda": tcuda,
            "host_cpu": cpu, "python": platform.python_version(), "versions": versions()}


def eval_prompts(path):
    rows = [json.loads(l)["prompt"] for l in open(path)]
    return rows[EVAL_ROWS[0]:EVAL_ROWS[1]]


# ------------------------------------------------------------------------------------------------ build
def build(variant, prompts, out_dir, recipe_mode="fixed"):
    """quantize.build() without Modal: same model load, recipes, calibration and save call.
    recipe_mode "published" reproduces the 2026-10-26 checkpoints (plain oneshot, embedding pruned for Wanda and
    SparseGPT); "fixed" keeps the tied embedding dense (quantize.oneshot_dense_embedding)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    q = load_quantize_funcs(prompts)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    info = {"variant": variant, "model": MODEL, "recipe_mode": recipe_mode, "env": env_block()}
    if variant == "sparse24_magnitude":
        info["layers_pruned"] = q["magnitude_24"](model)
    else:
        from llmcompressor import oneshot
        recipe, domain = q["recipe_for"](variant)
        if recipe_mode == "published":                     # the 2026-10-26 recipe: targets=["Linear"]
            recipe.targets = ["Linear"]
        info["recipe_targets"] = list(recipe.targets)
        kw = dict(dataset=q["calibration_set"](domain, tok), max_seq_length=q["SEQ_LEN"],
                  num_calibration_samples=q["N_CALIB"])
        if recipe_mode == "fixed":
            info["embedding_zero_frac_after_oneshot"] = q["oneshot_dense_embedding"](model, recipe, **kw)
        else:
            oneshot(model=model, recipe=recipe, **kw)
        info["calibration"] = {"domain": domain, "samples": q["N_CALIB"], "max_seq_length": q["SEQ_LEN"]}
    info.update(q["sparsity_report"](model))
    if recipe_mode == "fixed" and not (info["embedding_dense"] and abs(info["linear_sparsity"] - 0.5) < 0.01):
        raise SystemExit(f"fixed build failed: embedding_zero_frac {info['embedding_zero_frac']}, "
                         f"linear_sparsity {info['linear_sparsity']}")
    model.save_pretrained(out_dir, save_compressed=False)
    tok.save_pretrained(out_dir)
    info["build_s"] = round(time.time() - t0, 1)
    json.dump(info, open(os.path.join(out_dir, "rerun_build_info.json"), "w"), indent=2)
    print(json.dumps({k: info[k] for k in ("variant", "build_s", "linear_sparsity", "valid_2_4")}), flush=True)


# ------------------------------------------------------------------------------------------------ check
def check(ckpt, out):
    """Exact 2:4 along the input dimension in every pruned Linear; embedding (tied LM head) and norms dense;
    sha256 of every tensor, to compare the rebuild with the published checkpoint."""
    import torch
    from safetensors import safe_open
    rows, hashes, masks = {}, {}, {}
    for f in sorted(glob.glob(os.path.join(ckpt, "*.safetensors"))):
        with safe_open(f, "pt") as st:
            for k in st.keys():
                t = st.get_tensor(k)
                hashes[k] = hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]
                masks[k] = hashlib.sha256((t != 0).numpy().tobytes()).hexdigest()[:16]
                r = {"shape": list(t.shape), "dtype": str(t.dtype), "zero_frac": float((t == 0).float().mean())}
                if t.ndim == 2 and k.endswith("proj.weight"):
                    nz = (t.reshape(t.shape[0], -1, 4) != 0).sum(-1)      # groups of 4 along the input dim
                    r["groups_exactly_2"] = float((nz == 2).float().mean())
                    r["groups_at_most_2"] = float((nz <= 2).float().mean())
                rows[k] = r
    lin = {k: v for k, v in rows.items() if "groups_exactly_2" in v}
    summary = {"pruned_linear_tensors": len(lin),
               "all_linear_at_most_2_of_4": all(v["groups_at_most_2"] == 1.0 for v in lin.values()),
               "min_frac_groups_exactly_2": min((v["groups_exactly_2"] for v in lin.values()), default=None),
               "linear_zero_frac_mean": sum(v["zero_frac"] for v in lin.values()) / max(len(lin), 1),
               "dense_tensors_zero_frac_max": max((v["zero_frac"] for k, v in rows.items() if k not in lin),
                                                  default=None),
               "embed_tokens_zero_frac": next((v["zero_frac"] for k, v in rows.items() if "embed_tokens" in k), None),
               "lm_head_tensor_present": any("lm_head" in k for k in rows)}
    json.dump({"ckpt": os.path.basename(ckpt.rstrip("/")), "summary": summary, "tensors": rows, "sha256_16": hashes, "mask_sha256_16": masks},
              open(out, "w"), indent=2)
    print(json.dumps(summary), flush=True)


# ------------------------------------------------------------------------------------------------ perplexity
def ppl(ckpt, dtype, out):
    """lm-eval's wikitext task (rolling windows over the 62 test documents) on the Transformers backend, with the
    published run's settings: add_bos_token=True, max length 4,096, seed 1234."""
    od = f"/tmp/{os.environ.get('USER', 'u')}/lmeval_{os.getpid()}"
    args = f"pretrained={ckpt},dtype={dtype},add_bos_token=True,max_length=4096"
    cmd = ["lm_eval", "--model", "hf", "--model_args", args, "--tasks", "wikitext", "--batch_size", "4",
           "--seed", "1234", "--output_path", od, "--device", "cuda:0"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    res = {"ckpt": os.path.basename(ckpt.rstrip("/")), "dtype": dtype, "cmd": " ".join(cmd), "returncode": r.returncode,
           "wall_s": round(time.time() - t0, 1), "stderr_tail": r.stderr[-2500:] if r.returncode else ""}
    for f in pathlib.Path(od).rglob("results_*.json"):
        m = json.load(open(f))["results"]["wikitext"]
        res["metrics"] = {k: v for k, v in m.items() if isinstance(v, (int, float))}
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps({"ckpt": res["ckpt"], "dtype": dtype,
                      "word_ppl": res.get("metrics", {}).get("word_perplexity,none")}), flush=True)


# ------------------------------------------------------------------------------------------------ activations
def acts(ckpt, prompts, out, n_prompts=4):
    """Per decoder layer: max |hidden state| and the max |input| of mlp.down_proj (the GeGLU product), in BF16 and
    FP16, on the first held-out RAG prompts; non-finite counts and the first layer that overflows FP16."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt)
    texts = eval_prompts(prompts)[:n_prompts]
    res = {"ckpt": os.path.basename(ckpt.rstrip("/")), "prompts": n_prompts, "fp16_max": FP16_MAX, "by_dtype": {}}
    logits = {}
    for dname in ("bfloat16", "float16"):
        model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=getattr(torch, dname)).cuda().eval()
        layers = model.model.layers
        down = [0.0] * len(layers)
        down_nonfinite = [0] * len(layers)

        def hook(i):
            def f(_m, inp):
                x = inp[0].float()
                fin = torch.isfinite(x)
                down_nonfinite[i] += int((~fin).sum())
                down[i] = max(down[i], float(x[fin].abs().max()) if fin.any() else float("inf"))
            return f
        hs = [l.mlp.down_proj.register_forward_pre_hook(hook(i)) for i, l in enumerate(layers)]
        hid = [0.0] * (len(layers) + 1)
        hid_nonfinite = [0] * (len(layers) + 1)
        lg = []
        with torch.inference_mode():
            for t in texts:
                ids = tok(t, return_tensors="pt", add_special_tokens=True).to("cuda")
                o = model(**ids, output_hidden_states=True)
                for j, h in enumerate(o.hidden_states):
                    h = h.float()
                    fin = torch.isfinite(h)
                    hid_nonfinite[j] += int((~fin).sum())
                    hid[j] = max(hid[j], float(h[fin].abs().max()) if fin.any() else float("inf"))
                lg.append(o.logits[0].float().cpu())
        for h_ in hs:
            h_.remove()
        logits[dname] = lg
        first_over = next((i for i, v in enumerate(down) if v > FP16_MAX), None)
        res["by_dtype"][dname] = {
            "hidden_max_abs": hid, "hidden_nonfinite": hid_nonfinite,
            "down_proj_input_max_abs": down, "down_proj_input_nonfinite": down_nonfinite,
            "max_down_proj_input": max(down), "max_hidden": max(hid),
            "first_layer_down_input_over_fp16_max": first_over,
            "logits_nonfinite": int(sum((~torch.isfinite(x)).sum() for x in lg))}
        del model
        torch.cuda.empty_cache()
    a = torch.cat([x.argmax(-1) for x in logits["bfloat16"]])
    b = torch.cat([x.argmax(-1) for x in logits["float16"]])
    res["fp16_vs_bf16_top1_agreement"] = float((a == b).float().mean())
    json.dump(res, open(out, "w"), indent=2)
    bd = res["by_dtype"]
    print(json.dumps({"ckpt": res["ckpt"], "bf16_max_down_in": round(bd["bfloat16"]["max_down_proj_input"]),
                      "bf16_max_hidden": round(bd["bfloat16"]["max_hidden"]),
                      "fp16_nonfinite_hidden": sum(bd["float16"]["hidden_nonfinite"]),
                      "fp16_first_over": bd["float16"]["first_layer_down_input_over_fp16_max"],
                      "top1_fp16_vs_bf16": round(res["fp16_vs_bf16_top1_agreement"], 3)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["build", "check", "ppl", "acts"])
    ap.add_argument("--variant")
    ap.add_argument("--recipe", choices=["published", "fixed"], default="fixed")
    ap.add_argument("--prompts")
    ap.add_argument("--ckpt")
    ap.add_argument("--dtype", choices=["bfloat16", "float16"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.what == "build":
        build(a.variant, a.prompts, a.out, a.recipe)
    elif a.what == "check":
        check(a.ckpt, a.out)
    elif a.what == "ppl":
        ppl(a.ckpt, a.dtype, a.out)
    else:
        acts(a.ckpt, a.prompts, a.out)
