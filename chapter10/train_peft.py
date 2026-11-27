"""Fine-tune a generator with LoRA, QLoRA, DoRA, PiSSA or full fine-tuning (Chapter 10).

Writes the adapter or model to RUNS/<run>/ and RUNS/<run>/train_metrics.json with peak memory, tokens/s,
wall time, GPU-hours and cost. Full fine-tuning on two GPUs is launched with accelerate + FSDP (see modal_app.py).
"""
import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from common import DATA, MODELS, RUNS, Timer, save_json, NO_SYSTEM_ROLE, for_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-2b")
    ap.add_argument("--method", choices=["lora", "qlora", "full", "dora", "pissa"], default="lora")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--rslora", action="store_true")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mix-general", type=float, default=0.0, help="fraction of general instruction data")
    ap.add_argument("--train-file", default=f"{DATA}/train.jsonl")
    ap.add_argument("--gpu", default="H100")
    ap.add_argument("--n-gpus", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--run", default=None)
    ap.add_argument("--max-examples", type=int, default=0, help="use only the first N training examples (0 = all)")
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after N optimizer steps (-1 = full epochs)")
    a = ap.parse_args()

    run = a.run or f"{a.model}-{a.method}-r{a.rank}{'-rs' if a.rslora else ''}-s{a.seed}-{a.gpu}"
    out = f"{RUNS}/{run}"
    mid = MODELS[a.model]
    # BF16 matmuls need compute capability 8.0 (A100, H100). is_bf16_supported() also returns True on the T4 (7.5),
    # where BF16 GEMMs then fail, so decide on the capability: the T4 trains in FP16.
    bf16 = torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if bf16 else torch.float16
    lr = a.lr or (2e-5 if a.method == "full" else 1e-4)

    tok = AutoTokenizer.from_pretrained(mid)
    quant = None
    if a.method == "qlora":
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=dtype)
    VL = a.model in ("qwen3-vl-8b", "gemma-4-e2b")      # vision-language models: text-only LoRA on the language model
    if VL:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(mid, dtype=dtype, quantization_config=quant,
                                                            attn_implementation="sdpa")
    else:
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=dtype, quantization_config=quant,
                                                     attn_implementation="sdpa")

    ds = load_dataset("json", data_files=a.train_file, split="train")
    if a.max_examples:
        ds = ds.select(range(min(a.max_examples, len(ds))))
    if "answer" in ds.column_names:   # train/eval rows store the target separately: append it as the assistant turn
        ds = ds.map(lambda e: {"messages": e["messages"] + [{"role": "assistant", "content": e["answer"]}]})
    if a.mix_general > 0:
        gen = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=a.seed)
        gen = gen.select(range(int(len(ds) * a.mix_general))).map(lambda e: {"messages": e["messages"]})
        from datasets import concatenate_datasets
        ds = concatenate_datasets([ds.select_columns(["messages"]), gen.select_columns(["messages"])]).shuffle(a.seed)
    else:
        ds = ds.select_columns(["messages"])
    if a.model in NO_SYSTEM_ROLE:
        ds = ds.map(lambda e: {"messages": for_model(a.model, e["messages"])})

    print("first training example:", json.dumps(tok.apply_chat_template(ds[0]["messages"], tokenize=False)[-400:]),
          flush=True)
    peft_cfg = None
    if a.method != "full":
        peft_cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, target_modules=(r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
                                                              if VL else "all-linear"),
                              task_type="CAUSAL_LM", use_rslora=a.rslora, use_dora=(a.method == "dora"),
                              init_lora_weights=("pissa_niter_16" if a.method == "pissa" else True))

    cfg = SFTConfig(output_dir=out, num_train_epochs=a.epochs, learning_rate=lr, lr_scheduler_type="cosine",
                    warmup_steps=0.03, per_device_train_batch_size=8 // a.n_gpus if a.n_gpus > 1 else 8,
                    gradient_accumulation_steps=2, gradient_checkpointing=True, bf16=bf16, fp16=not bf16,
                    max_length=a.max_len, packing=True, assistant_only_loss=False, completion_only_loss=None,
                    logging_steps=10, save_strategy="no", report_to=[], seed=a.seed, max_steps=a.max_steps,
                    optim=("paged_adamw_8bit" if a.method == "qlora" else "adamw_torch_fused"))
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft_cfg)
    if not bf16:  # FP16 mixed precision (T4): trainable weights must be FP32 for the loss scaler, as in
        # peft.prepare_model_for_kbit_training; otherwise BF16 adapter weights fail in the gradient unscale step
        for p_ in trainer.model.parameters():
            if p_.requires_grad and p_.dtype != torch.float32:
                p_.data = p_.data.float()

    if a.method == "pissa":   # keep the initial adapter so that the trained one can be converted to plain LoRA
        trainer.model.peft_config["default"].init_lora_weights = True
        trainer.model.save_pretrained(f"{out}-pissa-init")
        trainer.model.peft_config["default"].init_lora_weights = "pissa_niter_16"
    torch.cuda.reset_peak_memory_stats()
    with Timer() as t:
        res = trainer.train()
    peak = torch.cuda.max_memory_allocated() / 1e9
    tokens = sum(len(x) for x in trainer.train_dataset["input_ids"]) * a.epochs
    if a.max_steps > 0:  # packed sequences of max_len tokens, global batch = per-device x accumulation x GPUs
        tokens = min(tokens, res.global_step * cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
                     * a.n_gpus * a.max_len)
    if a.method == "pissa":   # rank-2r LoRA on the original base weights (PEFT PiSSA conversion), servable by vLLM
        trainer.model.save_pretrained(out, path_initial_model_for_weight_conversion=f"{out}-pissa-init")
    else:
        trainer.save_model(out)
    tok.save_pretrained(out)
    gpu_hours = t.s / 3600 * a.n_gpus
    m = {"run": run, "model": a.model, "method": a.method, "rank": a.rank, "rslora": a.rslora, "gpu": a.gpu,
         "n_gpus": a.n_gpus, "seed": a.seed, "lr": lr, "max_examples": a.max_examples, "max_steps": a.max_steps,
         "steps": res.global_step, "peak_mem_gb": peak, "train_seconds": t.s,
         "tokens": tokens, "tokens_per_s": tokens / t.s, "gpu_hours": gpu_hours,
         "final_loss": res.training_loss,
         "trainable_params": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)}
    save_json(m, f"{out}/train_metrics.json")
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
