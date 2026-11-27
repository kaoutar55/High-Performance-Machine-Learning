"""Chapter 11 case study, TPU rows (Code 11.3). Run in a Colab notebook with a TPU runtime (see ch11_tpu_colab.ipynb).

Path A (TorchTPU): PyTorch model on device "tpu", compiled with torch.compile(backend="tpu"), inputs padded to 512.
Path B (JAX): KerasHub Gemma with the JAX backend.
Each path runs twice: default matmul precision and highest precision, for the parity comparison.
Writes results/case_study_tpu.json and out/*.npy; copy them back next to the other results.
HF_TOKEN must be set from Colab's secret store (userdata.get("HF_TOKEN")); never paste it into a cell.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import MODEL, MAX_PROMPT, NEW_TOKENS, PARITY_POSITIONS, load_prompts, chat, environment


def torchtpu(ref=None, precision="default"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if precision == "highest":
        torch.set_float32_matmul_precision("highest")   # affects fp32 inputs; BF16 weights are unaffected
    device = torch.device("tpu")
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
    dtype = torch.float32 if precision == "highest" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(device).eval()
    step = torch.compile(model, backend="tpu")
    prompts = [chat(tok, p) for p in load_prompts()]
    sync = torch.accelerator.synchronize

    def fwd(ids):
        with torch.no_grad():
            return step(input_ids=ids).logits

    ids = tok(prompts[:1], return_tensors="pt", padding="max_length", max_length=MAX_PROMPT, truncation=True).input_ids.to(device)
    t = time.perf_counter(); fwd(ids); sync(); cold = time.perf_counter() - t
    t = time.perf_counter(); fwd(ids); sync(); ttft = time.perf_counter() - t
    res = {"compile_s": round(cold - ttft, 1), "ttft_ms": round(ttft * 1e3, 1), "fallback_ops": None,
           "env": environment({"precision": precision})}
    # time per output token: generate() with a static cache (fixed shapes), same method as the GPU rows
    try:
        model.generation_config.cache_implementation = "static"
        def gen(n):
            with torch.no_grad():
                model.generate(input_ids=ids, max_new_tokens=n, min_new_tokens=n, do_sample=False)
            sync()
        gen(NEW_TOKENS)                                   # compile and warm up
        t = time.perf_counter(); gen(1); t1 = time.perf_counter() - t
        t = time.perf_counter(); gen(NEW_TOKENS); tn = time.perf_counter() - t
        res["tpot_b1_ms"] = round((tn - t1) / (NEW_TOKENS - 1) * 1e3, 2)
    except Exception as e:                                # record the failure; it is a finding in its own right
        res["tpot_error"] = repr(e)[:300]
    if ref:
        logits = []
        for pid, toks in zip(ref["prompt_ids"], ref["tokens"]):
            full = torch.tensor([pid + toks[:PARITY_POSITIONS]], device=device)
            lg = fwd(full)[0, len(pid) - 1: len(pid) - 1 + PARITY_POSITIONS].float().cpu().numpy()
            logits.append(lg)
        np.save(f"out/tpu_torchtpu_{precision}.npy", np.stack(logits))
    return res


def jax_keras(ref=None, precision="default"):
    os.environ["KERAS_BACKEND"] = "jax"
    import jax, keras_hub
    jax.config.update("jax_default_matmul_precision", "highest" if precision == "highest" else None)
    lm = keras_hub.models.GemmaCausalLM.from_preset("gemma_instruct_2b_en",
                                                    dtype="bfloat16" if precision == "default" else "float32")
    prompts = load_prompts()
    lm.compile(sampler="greedy")
    t = time.perf_counter(); lm.generate(prompts[0], max_length=MAX_PROMPT + NEW_TOKENS); cold = time.perf_counter() - t
    t = time.perf_counter(); lm.generate(prompts[0], max_length=MAX_PROMPT + NEW_TOKENS); warm = time.perf_counter() - t
    lm.generate(prompts[0], max_length=MAX_PROMPT + 1)
    t = time.perf_counter(); lm.generate(prompts[0], max_length=MAX_PROMPT + 1); t1 = time.perf_counter() - t
    res = {"compile_s": round(cold - warm, 1), "ttft_ms": round(t1 * 1e3, 1),
           "tpot_b1_ms": round((warm - t1) / (NEW_TOKENS - 1) * 1e3, 2), "env": environment({"precision": precision})}
    if ref:
        logits = []
        for pid, toks in zip(ref["prompt_ids"], ref["tokens"]):
            x = np.array([pid + toks[:PARITY_POSITIONS]], dtype=np.int32)
            out = lm.backbone({"token_ids": x, "padding_mask": np.ones_like(x)})
            lg = lm.backbone.token_embedding(out, reverse=True)
            logits.append(np.asarray(lg[0, len(pid) - 1: len(pid) - 1 + PARITY_POSITIONS], dtype=np.float32))
        np.save(f"out/tpu_jax_{precision}.npy", np.stack(logits))
    return res


if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    ref = None
    if os.path.exists("ref/ref_ids.json"):
        ref = json.load(open("ref/ref_ids.json")); ref["tokens"] = np.load("ref/ref_tokens.npy").tolist()
    which = sys.argv[1] if len(sys.argv) > 1 else "torchtpu"
    out = {}
    for prec in ("default", "highest"):
        out[f"tpu_{which}_{prec}"] = (torchtpu if which == "torchtpu" else jax_keras)(ref, prec)
    json.dump(out, open(f"case_study_tpu_{which}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
