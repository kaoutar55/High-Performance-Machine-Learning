"""FP32 CPU reference for parity: greedy tokens and teacher-forced logits at the first 32 generated positions.

Writes ref_cpu_fp32.npy (logits), ref_tokens.npy (128 greedy tokens per prompt) and ref_ids.json (prompt ids),
which every other platform reuses so that all logits are computed on the same token sequences.
Runs on a CPU in about an hour for 32 prompts; run it once on Modal (cpu=16) or locally.
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import MODEL, MAX_PROMPT, NEW_TOKENS, PARITY_POSITIONS, load_prompts, chat, environment
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(prompts_path=None, out="ref"):
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    ids_all, toks, logits = [], [], []
    for p in load_prompts(prompts_path):
        ids = tok(chat(tok, p), return_tensors="pt", truncation=True, max_length=MAX_PROMPT).input_ids
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=NEW_TOKENS, do_sample=False, min_new_tokens=NEW_TOKENS)
            full = g[:, : ids.shape[1] + PARITY_POSITIONS]
            lg = model(full).logits[0, ids.shape[1] - 1: ids.shape[1] - 1 + PARITY_POSITIONS]
        ids_all.append(ids[0].tolist()); toks.append(g[0, ids.shape[1]:].numpy()); logits.append(lg.float().numpy())
    np.save(f"{out}/ref_cpu_fp32.npy", np.stack(logits)); np.save(f"{out}/ref_tokens.npy", np.stack(toks))
    json.dump({"prompt_ids": ids_all, "env": environment()}, open(f"{out}/ref_ids.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
