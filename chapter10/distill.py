"""Distillation arms for Chapter 10: teacher = Llama-3.1-8B-Instruct + HotpotQA adapter (merged).

  --arm sft     student fine-tuned on the reference answers (train.jsonl)
  --arm seqkd   student fine-tuned on teacher-generated answers (teacher_answers.jsonl, made with --make-teacher-data)
  --arm tokenkd token-level KD with forward KL, teacher run online (same tokenizer required)
  --arm gkd     on-policy distillation with TRL GKDTrainer (reverse KL, student samples)
Students: llama-3.2-1b (all arms) and gemma-2b (sft and seqkd only: different tokenizer).
"""
import argparse
import json

import torch
import torch.nn.functional as F

from common import DATA, MODELS, RUNS, Timer, for_model, save_json


def kd_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    """Logits: (batch, seq, vocab); labels: (batch, seq), -100 = ignore."""
    mask = labels != -100
    s = F.log_softmax(student_logits[mask] / T, dim=-1)
    t = F.log_softmax(teacher_logits[mask] / T, dim=-1)
    soft = F.kl_div(s, t, log_target=True, reduction="batchmean") * T * T
    hard = F.cross_entropy(student_logits[mask], labels[mask])
    return alpha * soft + (1 - alpha) * hard


def make_teacher_data(teacher_dir, max_examples=0):
    from vllm import LLM, SamplingParams
    rows = [json.loads(l) for l in open(f"{DATA}/train.jsonl")][:max_examples or None]
    llm = LLM(model=teacher_dir, max_model_len=4096, seed=0)
    outs = llm.chat([r["messages"] for r in rows], SamplingParams(temperature=0.0, max_tokens=64))
    with open(f"{DATA}/teacher_answers.jsonl", "w") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps({"messages": r["messages"] + [{"role": "assistant", "content": o.outputs[0].text}]})
                    + "\n")


def to_chat(rows):
    return [{"messages": r["messages"] + [{"role": "assistant", "content": r["answer"]}]} for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["sft", "seqkd", "tokenkd", "gkd"], required=True)
    ap.add_argument("--student", default="llama-3.2-1b")
    ap.add_argument("--teacher-run", default="llama-3.1-8b-lora-r16-s0-H100")
    ap.add_argument("--make-teacher-data", action="store_true")
    ap.add_argument("--gpu", default="H100")
    ap.add_argument("--max-examples", type=int, default=0, help="first N training examples only (0 = all)")
    a = ap.parse_args()
    teacher_dir = f"{RUNS}/{a.teacher_run}-merged"
    if a.make_teacher_data:
        make_teacher_data(teacher_dir, a.max_examples); return

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sid = MODELS[a.student]
    tok = AutoTokenizer.from_pretrained(sid)
    if tok.pad_token is None:                  # Llama-3.2-1B has no pad token; GKDTrainer requires one
        tok.pad_token = tok.eos_token
    student = AutoModelForCausalLM.from_pretrained(sid, dtype=torch.bfloat16)
    rows = [{**r, "messages": for_model(a.student, r["messages"])}
            for r in [json.loads(l) for l in open(f"{DATA}/train.jsonl")][:a.max_examples or None]]
    out = f"{RUNS}/distill-{a.student}-{a.arm}"

    with Timer() as t:
        if a.arm in ("sft", "seqkd", "tokenkd"):
            from trl import SFTConfig, SFTTrainer
            data = (to_chat(rows) if a.arm != "seqkd"
                    else [{"messages": for_model(a.student, json.loads(l)["messages"])}
                          for l in open(f"{DATA}/teacher_answers.jsonl")][:a.max_examples or None])
            cfg = SFTConfig(output_dir=out, num_train_epochs=2, learning_rate=2e-5, per_device_train_batch_size=8,
                            gradient_accumulation_steps=2, bf16=True, max_length=1024, save_strategy="no",
                            report_to=[], gradient_checkpointing=True)
            if a.arm == "tokenkd":
                teacher = AutoModelForCausalLM.from_pretrained(teacher_dir, dtype=torch.bfloat16).cuda().eval()

                class KDTrainer(SFTTrainer):
                    def compute_loss(self, model, inputs, return_outputs=False, **kw):
                        labels = inputs["input_ids"].clone()
                        if "attention_mask" in inputs:
                            labels[inputs["attention_mask"] == 0] = -100
                        so = model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
                        with torch.no_grad():
                            to = teacher(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
                        loss = kd_loss(so.logits[:, :-1], to.logits[:, :-1], labels[:, 1:])
                        return (loss, so) if return_outputs else loss
                tr = KDTrainer(model=student, args=cfg, train_dataset=Dataset.from_list(data), processing_class=tok)
            else:
                tr = SFTTrainer(model=student, args=cfg, train_dataset=Dataset.from_list(data), processing_class=tok)
            tr.train()
        else:
            from trl.experimental.gkd import GKDConfig, GKDTrainer
            teacher = AutoModelForCausalLM.from_pretrained(teacher_dir, dtype=torch.bfloat16)
            prompts = Dataset.from_list(to_chat(rows[:5000]))
            args = GKDConfig(output_dir=out, lmbda=1.0, beta=1.0, temperature=0.9, max_new_tokens=128,
                             learning_rate=1e-5, per_device_train_batch_size=8, bf16=True, save_strategy="no",
                             report_to=[], num_train_epochs=1)
            tr = GKDTrainer(model=student, teacher_model=teacher, args=args, processing_class=tok,
                            train_dataset=prompts)
            tr.train()
    tr.save_model(out); tok.save_pretrained(out)
    save_json({"arm": a.arm, "student": a.student, "train_seconds": t.s, "gpu_hours": t.s / 3600}, f"{out}/train_metrics.json")


if __name__ == "__main__":
    main()
