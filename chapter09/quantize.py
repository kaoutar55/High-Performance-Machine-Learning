"""Chapter 9 case study, step 1: build the compressed checkpoints with llm-compressor on Modal.

Every Gemma-2B-IT variant is built here, so all Gemma schemes share one controlled pipeline. For
Llama-3.1-8B-Instruct the case study uses Red Hat's published FP8, INT8 and INT4 checkpoints, and this script
builds only what has to be controlled: the calibration-data study (the same W4A16 GPTQ recipe calibrated on
three different datasets), an AWQ checkpoint, an FP8 checkpoint with calibrated KV-cache scales, and a 2:4
pruned model.

    modal run chapter09/quantize.py                          # every variant of both models
    modal run chapter09/quantize.py --models gemma --variants w4a16_gptq_rag,sparse24_wanda

Checkpoints go to the Modal volume "hpml-ch09-ckpt" as /ckpt/<model>-<variant>. llm-compressor runs in its
own image because it pins different torch, transformers and compressed-tensors versions than vLLM 0.32.

Calibration data (512 samples each, never the evaluation prompts):
  rag        rows 0-511 of the Chapter 8 RAG prompts (the text the server prefills)
  ultrachat  512 chat conversations from HuggingFaceH4/ultrachat_200k, with the model's chat template
  wikitext   512 chunks of wikitext-2 raw text, the usual "generic" calibration set
The held-out evaluation prompts are rows 800-999 of the same file.
"""
import json
import os
import pathlib
import time

import modal

HERE = pathlib.Path(__file__).resolve().parent
LLMC_VERSION = os.environ.get("CH09_LLMC", "0.14.0.1")

MODELS = {"gemma": "google/gemma-2b-it", "llama": "meta-llama/Llama-3.1-8B-Instruct"}
VARIANTS = {
    "gemma": ["fp8_dynamic", "w8a8_int8", "w4a16_gptq_rag", "w4a16_gptq_wikitext", "w4a16_awq_rag",
              "sparse24_magnitude", "sparse24_wanda", "sparse24_sparsegpt"],
    "llama": ["w4a16_gptq_rag", "w4a16_gptq_ultrachat", "w4a16_gptq_wikitext", "w4a16_awq_rag",
              "fp8_dynamic_kv", "sparse24_sparsegpt"],
}
N_CALIB, SEQ_LEN = 512, 1024

ckpt_vol = modal.Volume.from_name("hpml-ch09-ckpt", create_if_missing=True)
hf_cache = modal.Volume.from_name("hpml-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("hpml-ch08-data", create_if_missing=True)

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install(f"llmcompressor=={LLMC_VERSION}", "datasets")
         .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"}))
app = modal.App("hpml-ch09-quantize")


# ------------------------------------------------------------------------------------------------ calibration
def calibration_set(domain, tok):
    from datasets import Dataset, load_dataset
    rows = []
    if domain == "rag":
        prompts = [json.loads(l)["prompt"] for l in open("/data/prompts_single.jsonl")][:N_CALIB]
        rows = [tok(p, add_special_tokens=True, max_length=SEQ_LEN, truncation=True) for p in prompts]
    elif domain == "ultrachat":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{N_CALIB}]")
        for ex in ds:
            text = tok.apply_chat_template(ex["messages"], tokenize=False)
            rows.append(tok(text, add_special_tokens=False, max_length=SEQ_LEN, truncation=True))
    elif domain == "wikitext":
        text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                                                    split="train")["text"] if t.strip())
        ids = tok(text, add_special_tokens=False)["input_ids"]
        step = 512                                         # same length as the RAG prompts
        for i in range(N_CALIB):
            chunk = [tok.bos_token_id] + ids[i * step:(i + 1) * step - 1]
            rows.append({"input_ids": chunk, "attention_mask": [1] * len(chunk)})
    else:
        raise ValueError(domain)
    return Dataset.from_list([{"input_ids": r["input_ids"], "attention_mask": r["attention_mask"]} for r in rows])


# ------------------------------------------------------------------------------------------------ recipes
def recipe_for(variant):
    from llmcompressor.modifiers.quantization import QuantizationModifier
    try:
        from llmcompressor.modifiers.gptq import GPTQModifier
    except ImportError:                                    # older layout
        from llmcompressor.modifiers.quantization import GPTQModifier
    if variant == "fp8_dynamic":                           # data-free: per-channel FP8 W, dynamic per-token FP8 A
        return QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]), None
    if variant == "fp8_dynamic_kv":                        # plus static per-tensor FP8 KV scales from calibration
        kv = {"num_bits": 8, "type": "float", "strategy": "tensor", "dynamic": False, "symmetric": True}
        return QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"],
                                    kv_cache_scheme=kv), "rag"
    if variant == "w8a8_int8":                             # SmoothQuant moves activation outliers into the weights
        from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
        return [SmoothQuantModifier(smoothing_strength=0.8),
                GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])], "rag"
    if variant.startswith("w4a16_gptq_"):                  # symmetric INT4, group 128
        return GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]), variant.split("_")[-1]
    if variant == "w4a16_awq_rag":                         # asymmetric INT4 with zero points, group 128
        from llmcompressor.modifiers.transform.awq import AWQModifier
        return [AWQModifier(), QuantizationModifier(targets=["Linear"], scheme="W4A16_ASYM",
                                                    ignore=["lm_head"])], "rag"
    if variant in ("sparse24_wanda", "sparse24_sparsegpt"):
        try:
            from llmcompressor.modifiers.pruning import WandaPruningModifier, SparseGPTModifier
        except ImportError:                                # older layout
            from llmcompressor.modifiers.pruning.wanda import WandaPruningModifier
            from llmcompressor.modifiers.obcq import SparseGPTModifier
        cls = WandaPruningModifier if variant.endswith("wanda") else SparseGPTModifier
        return cls(sparsity=0.5, mask_structure="2:4", targets=["Linear"], ignore=["lm_head"]), "rag"
    raise ValueError(variant)


def magnitude_24(model):
    """2:4 magnitude pruning: in every group of 4 consecutive input weights keep the 2 largest |w|."""
    import torch
    n = 0
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and "lm_head" not in name:
            w = m.weight.data
            g = w.reshape(-1, 4)
            drop = g.abs().argsort(dim=1)[:, :2]               # indices of the 2 smallest in each group
            g.scatter_(1, drop, 0)
            n += 1
    return n


def sparsity_report(model):
    import torch
    zeros = total = 0
    ok = True
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and "lm_head" not in name:
            w = m.weight.data
            zeros += (w == 0).sum().item(); total += w.numel()
            ok &= bool(((w.reshape(-1, 4) != 0).sum(dim=1) <= 2).all())
    return {"linear_sparsity": zeros / total, "valid_2_4": ok}


# ------------------------------------------------------------------------------------------------ worker
@app.function(image=image, gpu="A100-80GB", memory=65536, timeout=4 * 3600,
              volumes={"/ckpt": ckpt_vol, "/cache": hf_cache, "/data": data_vol},
              secrets=[modal.Secret.from_name("huggingface")])
def build(model_key: str, variant: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out_dir = f"/ckpt/{model_key}-{variant}"
    if os.path.exists(os.path.join(out_dir, "config.json")):
        return {"variant": variant, "skipped": "exists", "path": out_dir}
    t0 = time.time()
    model_id = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    info = {"variant": variant, "model": model_id}
    if variant == "sparse24_magnitude":
        info["layers_pruned"] = magnitude_24(model)
        save_compressed = False
    else:
        from llmcompressor import oneshot
        recipe, domain = recipe_for(variant)
        kwargs = {}
        if domain:
            kwargs = dict(dataset=calibration_set(domain, tok), max_seq_length=SEQ_LEN,
                          num_calibration_samples=N_CALIB)
            info["calibration"] = {"domain": domain, "samples": N_CALIB, "max_seq_length": SEQ_LEN}
        oneshot(model=model, recipe=recipe, **kwargs)
        # vLLM 0.32 rejects checkpoints that declare sparsity, so pruned models are saved as dense weights
        save_compressed = not variant.startswith("sparse24")
    if variant.startswith("sparse24"):
        info.update(sparsity_report(model))
    model.save_pretrained(out_dir, save_compressed=save_compressed)
    tok.save_pretrained(out_dir)
    info["build_s"] = round(time.time() - t0, 1)
    info["size_gb"] = round(sum(p.stat().st_size for p in pathlib.Path(out_dir).glob("*.safetensors")) / 1e9, 3)
    json.dump(info, open(os.path.join(out_dir, "build_info.json"), "w"), indent=2)
    ckpt_vol.commit()
    return info


@app.local_entrypoint()
def main(models: str = "", variants: str = ""):
    sel_m = [m for m in (models.split(",") if models else MODELS)]
    sel_v = set(v for v in variants.split(",") if v)
    jobs = [(m, v) for m in sel_m for v in VARIANTS[m] if not sel_v or v in sel_v]
    results = list(build.starmap(jobs, return_exceptions=True))
    out = HERE / "results"; out.mkdir(exist_ok=True)
    rows = []
    for (m, v), r in zip(jobs, results):
        rows.append(r if isinstance(r, dict) else {"model": MODELS[m], "variant": v, "error": repr(r)[-2000:]})
        print(m, v, rows[-1].get("error", rows[-1].get("skipped", f"{rows[-1].get('size_gb')} GB "
                                                                    f"in {rows[-1].get('build_s')} s")))
    path = out / "build_info.json"
    old = json.loads(path.read_text()) if path.exists() else []
    keep = [r for r in old if (r.get("model"), r.get("variant")) not in {(MODELS[m], v) for m, v in jobs}]
    path.write_text(json.dumps(keep + rows, indent=2))
