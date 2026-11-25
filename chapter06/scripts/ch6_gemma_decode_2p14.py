"""Chapter 6 (PyTorch 2.14.1 / Triton 3.8.0): one Gemma-2B decode step with and without the chapter's Triton
RMSNorm patched into every RMSNorm module of the model.

The decode measurement is Chapter 4's: case_study/experiments.py loads google/gemma-2b-it in FP16 with SDPA
attention, builds the 512-token RAG prompt, and times greedy generate() with the dynamic cache;
decode ms per token = (t(20 new tokens) - t(1 new token)) / 19, each t the median of 10 calls after 2 warm-up calls.

The RMSNorm kernel is the notebook's rmsnorm_fwd_kernel, unchanged. Gemma's RMSNorm scales by (1 + w), so the patch
passes the precomputed FP32 tensor (1 + w) as the kernel's weight. Before timing, each patched module is checked
against its original forward on a random input; a mismatch stops the run. The MLP activation is not patched:
Gemma's MLP is GeGLU with the tanh GELU, and the chapter has no kernel for it (its fused kernel is QuickGELU).

    python ch6_gemma_decode_2p14.py --order ab --out d1.json   # a = unpatched first, b = patched first
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CASE = os.environ.get("CH6_CASE_STUDY", os.path.join(HERE, "..", "..", "case_study"))
sys.path.insert(0, CASE)
import ch6_measure_2p14 as cm  # noqa: E402
import matmul_sizes_2p14 as ms  # noqa: E402


def main(out, order):
    import torch
    import transformers
    import experiments as ex            # case_study: _generator, _prompt, _cuda_time
    import experiment_config as cfg
    state_before = cm.gpu_state()
    ns = cm.load_defs()
    tok, model = ex._generator()
    inputs = ex._prompt(tok)
    norms = [(n, m) for n, m in model.named_modules() if type(m).__name__.endswith("RMSNorm")]
    originals = {}

    def patch():
        for name, m in norms:
            eps = float(getattr(m, "eps", getattr(m, "variance_epsilon", 1e-6)))
            w1 = (1.0 + m.weight.detach().float()).contiguous()
            orig = m.forward

            def fwd(x, _w=w1, _eps=eps):
                shp = x.shape
                return ns["rmsnorm"](x.reshape(-1, shp[-1]).contiguous(), _w, _eps).view(shp)
            probe = torch.randn(4, m.weight.shape[0], device="cuda", dtype=torch.float16)
            with torch.inference_mode():
                a, b = orig(probe), fwd(probe)
            if not torch.allclose(a.float(), b.float(), atol=1e-2, rtol=1e-2):
                raise SystemExit(f"{name}: Triton RMSNorm with (1 + w) does not match {type(m).__name__}.forward "
                                 f"(max abs err {(a.float() - b.float()).abs().max():.3e})")
            originals[name] = orig
            m.forward = fwd

    def unpatch():
        for name, m in norms:
            if name in originals:
                m.forward = originals.pop(name)

    def gen(n):
        return lambda: model.generate(**inputs, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                                      pad_token_id=tok.eos_token_id)

    def measure():
        gen(cfg.NEW_TOKENS)()                     # first call outside the timer (Triton JIT on the patched path)
        t1 = ex._cuda_time(gen(1))
        tn = ex._cuda_time(gen(cfg.NEW_TOKENS))
        with torch.inference_mode():
            ids = gen(cfg.NEW_TOKENS)()[0, inputs["input_ids"].shape[1]:].tolist()
            logits = model(**inputs).logits[0, -1].float()
        return {"t_first_token_s": t1, "t_all_tokens_s": tn,
                "decode_ms_per_token": (tn - t1) / (cfg.NEW_TOKENS - 1) * 1000}, ids, logits

    res = {"kind": "decode", "env": {**ms.env_info(" ".join(["python"] + sys.argv)),
                                     "transformers": transformers.__version__},
           "gpu_state_before": state_before, "model": cfg.MODELS["generator"], "dtype": cfg.DTYPE,
           "prompt_tokens": int(inputs["input_ids"].shape[1]), "new_tokens": cfg.NEW_TOKENS,
           "rmsnorm_modules": len(norms), "rmsnorm_class": sorted({type(m).__name__ for _, m in norms}),
           "order": order}
    runs = {}
    with torch.inference_mode():
        for label in (("unpatched", "patched") if order == "ab" else ("patched", "unpatched")):
            patch() if label == "patched" else unpatch()
            runs[label] = measure()
    unpatch()
    res["unpatched"], ids_u, lg_u = runs["unpatched"]
    res["patched"], ids_p, lg_p = runs["patched"]
    res["patched_over_unpatched"] = res["patched"]["decode_ms_per_token"] / res["unpatched"]["decode_ms_per_token"]
    res["same_greedy_tokens"] = ids_u == ids_p
    res["matching_greedy_tokens"] = sum(a == b for a, b in zip(ids_u, ids_p))
    res["last_logits_max_abs_diff"] = float((lg_u - lg_p).abs().max())
    res["gpu_state_after"] = cm.gpu_state()
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("order", "patched_over_unpatched", "same_greedy_tokens")} |
                     {"unpatched_ms": round(res["unpatched"]["decode_ms_per_token"], 2),
                      "patched_ms": round(res["patched"]["decode_ms_per_token"], 2)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--order", choices=["ab", "ba"], default="ab")
    a = ap.parse_args()
    main(a.out, a.order)
