"""Chapter 11 case study, NVIDIA reference rows (Modal).

  modal run h100/modal_reference.py --which torch   # H100 PyTorch eager and torch.compile (static cache)
  modal run h100/modal_reference.py --which vllm    # H100 vLLM 0.33.0
Run parity/cpu_reference.py first so that parity logits are computed on the reference token sequences.
Results are merged into ../../results/case_study.json and parity logits are written to ./out/.
Requires a Modal secret named "huggingface" holding HF_TOKEN (accepted Gemma licence).
"""
import json, os, sys, time
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("hpml-ch11-reference")
hf = modal.Volume.from_name("hf-cache", create_if_missing=True)
SECRETS = [modal.Secret.from_name("huggingface")]
torch_img = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("torch==2.14.1", "transformers>=5.16,<5.20", "accelerate", "datasets", "numpy")
             .add_local_file(os.path.join(HERE, "..", "common.py"), "/root/common.py"))
vllm_img = (modal.Image.debian_slim(python_version="3.11").pip_install("vllm==0.33.0", "datasets")
            .add_local_file(os.path.join(HERE, "..", "common.py"), "/root/common.py"))


def _timed(fn):
    import torch
    torch.cuda.synchronize(); t = time.perf_counter(); r = fn(); torch.cuda.synchronize()
    return r, time.perf_counter() - t


@app.function(image=torch_img, gpu="H100", secrets=SECRETS, volumes={"/root/.cache/huggingface": hf}, timeout=3600)
def torch_rows(ref_ids_json: str = ""):
    import numpy as np, torch
    sys.path.insert(0, "/root")
    from common import MODEL, MAX_PROMPT, NEW_TOKENS, PARITY_POSITIONS, load_prompts, chat, environment
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
    prompts = [chat(tok, p) for p in load_prompts()]
    out = {}
    for mode in ("eager", "compile"):
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
        kw = dict(do_sample=False, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS)
        if mode == "compile":
            model.generation_config.cache_implementation = "static"
            model.forward = torch.compile(model.forward)
        def batch(n):
            return tok(prompts[:n], return_tensors="pt", padding="max_length", truncation=True,
                       max_length=MAX_PROMPT).to("cuda")
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _, cold = _timed(lambda: model.generate(**batch(1), **kw))          # includes compilation
            _, warm = _timed(lambda: model.generate(**batch(1), **kw))
            res = {}
            for n in (1, 16):
                b = batch(n)
                _, t1 = _timed(lambda: model.generate(**b, do_sample=False, max_new_tokens=1))
                _, tn = _timed(lambda: model.generate(**b, **kw))
                res[n] = (t1, (tn - t1) / (NEW_TOKENS - 1))
        out[f"h100_{mode}"] = {"compile_s": round(cold - warm, 2) if mode == "compile" else None,
                               "ttft_ms": round(res[1][0] * 1e3, 2), "tpot_b1_ms": round(res[1][1] * 1e3, 3),
                               "tpot_b16_ms": round(res[16][1] * 1e3, 3),
                               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "fallback_ops": 0,
                               "env": environment({"mode": mode})}
        if ref_ids_json:   # teacher-forced logits on the reference token sequences, for parity
            ref = json.loads(ref_ids_json); logits = []
            for ids, toks in zip(ref["prompt_ids"], ref["tokens"]):
                full = torch.tensor([ids + toks[:PARITY_POSITIONS]], device="cuda")
                with torch.no_grad():
                    lg = model(full).logits[0, len(ids) - 1: len(ids) - 1 + PARITY_POSITIONS]
                logits.append(lg.float().cpu().numpy())
            out[f"h100_{mode}"]["_logits"] = np.stack(logits).tobytes()
        del model; torch.cuda.empty_cache()
    return out


@app.function(image=vllm_img, gpu="H100", secrets=SECRETS, volumes={"/root/.cache/huggingface": hf}, timeout=3600)
def vllm_row():
    sys.path.insert(0, "/root")
    from common import MODEL, MAX_PROMPT, NEW_TOKENS, load_prompts, environment
    from vllm import LLM, SamplingParams
    llm = LLM(MODEL, dtype="bfloat16", max_model_len=2048, enable_prefix_caching=False)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
               for p in load_prompts()]
    one = SamplingParams(temperature=0, max_tokens=1)
    full = SamplingParams(temperature=0, max_tokens=NEW_TOKENS, min_tokens=NEW_TOKENS)
    llm.generate(prompts[:2], full)                                         # warm-up
    res = {}
    for n in (1, 16):
        t = time.perf_counter(); llm.generate(prompts[:n], one); t1 = time.perf_counter() - t
        t = time.perf_counter(); llm.generate(prompts[:n], full); tn = time.perf_counter() - t
        res[n] = (t1, (tn - t1) / (NEW_TOKENS - 1))
    return {"h100_vllm": {"ttft_ms": round(res[1][0] * 1e3, 2), "tpot_b1_ms": round(res[1][1] * 1e3, 3),
                          "tpot_b16_ms": round(res[16][1] * 1e3, 3), "fallback_ops": 0, "env": environment()}}


def _store(rows):
    import numpy as np
    sys.path.insert(0, os.path.join(HERE, ".."))
    from common import merge_json
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    res = os.path.join(HERE, "..", "..", "..", "results", "case_study.json")
    for k, v in rows.items():
        blob = v.pop("_logits", None)
        if blob is not None:
            np.save(os.path.join(HERE, "out", f"{k}.npy"), np.frombuffer(blob, dtype=np.float32).reshape(32, 32, -1))
        merge_json(res, k, v)
    print(json.dumps(rows, indent=1, default=str)[:2000])


@app.local_entrypoint()
def main(which: str = "all", ref: str = "../parity/ref/ref_ids.json"):
    ref_json = ""
    if os.path.exists(ref):
        import numpy as np
        r = json.load(open(ref))
        r["tokens"] = np.load(os.path.join(os.path.dirname(ref), "ref_tokens.npy")).tolist()
        ref_json = json.dumps(r)
    if which in ("all", "torch"):
        _store(torch_rows.remote(ref_json))
    if which in ("all", "vllm"):
        _store(vllm_row.remote())
