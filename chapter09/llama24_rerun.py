"""Chapter 9: rebuild of the Llama-3.1-8B-Instruct 2:4 SparseGPT checkpoint with the LM head kept dense.

Uses gemma24_rerun.build() (quantize.py's own recipe, calibration and save code) with the model id set to Llama, and
adds a `masks` check for the untied LM head and the 32-layer decoder.

    python llama24_rerun.py build --variant sparse24_sparsegpt --recipe fixed --prompts prompts_single.jsonl --out DIR
    python llama24_rerun.py masks --ckpt DIR --out masks.json [--expect fixed]
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemma24_rerun  # noqa: E402

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
N_LAYERS = 32
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def masks(ckpt, out, expect=None):
    """gemma24_rerun.check() plus: the 2:4 group test on lm_head.weight, lm_head and embed_tokens zero fractions,
    and the count of exactly-2:4 tensors per projection name and per layer (7 x 32 = 224 expected)."""
    import torch
    from safetensors import safe_open
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    gemma24_rerun.check(ckpt, tmp)
    base = json.load(open(tmp))
    os.unlink(tmp)
    rows = base["tensors"]
    lm = {}
    for f in sorted(glob.glob(os.path.join(ckpt, "*.safetensors"))):
        with safe_open(f, "pt") as st:
            if "lm_head.weight" in st.keys():
                t = st.get_tensor("lm_head.weight")
                nz = (t.reshape(t.shape[0], -1, 4) != 0).sum(-1)        # groups of 4 along the input dimension
                lm = {"shape": list(t.shape), "zero_frac": float((t == 0).float().mean()),
                      "groups_exactly_2": float((nz == 2).float().mean())}
    per_proj, per_layer = collections.Counter(), collections.Counter()
    for k, v in rows.items():
        m = re.match(r"model\.layers\.(\d+)\..*\.(\w+_proj)\.weight$", k)
        if m and v.get("groups_exactly_2") == 1.0:
            per_proj[m.group(2)] += 1
            per_layer[int(m.group(1))] += 1
    emb = rows.get("model.embed_tokens.weight", {}).get("zero_frac")
    summary = {**base["summary"], "lm_head": lm or "not stored (tied)", "lm_head_zero_frac": lm.get("zero_frac"),
               "lm_head_groups_exactly_2": lm.get("groups_exactly_2"), "embed_tokens_zero_frac": emb,
               "exact_2_4_per_projection": {p: per_proj.get(p, 0) for p in PROJ},
               "exact_2_4_per_layer": {str(i): per_layer.get(i, 0) for i in range(N_LAYERS)},
               "exact_2_4_decoder_tensors": sum(per_proj.values())}
    ok_fixed = (summary["exact_2_4_decoder_tensors"] == 7 * N_LAYERS and summary["min_frac_groups_exactly_2"] == 1.0
                and emb is not None and emb < 0.01 and lm and lm["zero_frac"] < 0.01)
    summary["passes_fixed_criteria"] = bool(ok_fixed)
    json.dump({"ckpt": os.path.basename(ckpt.rstrip("/")), "summary": summary, "tensors": rows,
               "sha256_16": base["sha256_16"], "mask_sha256_16": base["mask_sha256_16"]}, open(out, "w"), indent=2)
    print(json.dumps({k: summary[k] for k in ("exact_2_4_decoder_tensors", "min_frac_groups_exactly_2", "lm_head_zero_frac",
                                              "lm_head_groups_exactly_2", "embed_tokens_zero_frac", "passes_fixed_criteria")}),
          flush=True)
    if expect == "fixed" and not ok_fixed:
        raise SystemExit("mask check failed for a fixed build")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["build", "masks"])
    ap.add_argument("--variant", default="sparse24_sparsegpt")
    ap.add_argument("--recipe", choices=["published", "fixed"], default="fixed")
    ap.add_argument("--prompts")
    ap.add_argument("--ckpt")
    ap.add_argument("--expect", choices=["fixed"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.what == "build":
        gemma24_rerun.MODEL = MODEL                    # build() reads the module constant
        gemma24_rerun.build(a.variant, a.prompts, a.out, a.recipe)
    else:
        masks(a.ckpt, a.out, a.expect)
