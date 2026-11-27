"""Shared settings for the Chapter 10 case study."""
import json
import os
import time

MODELS = {
    "gemma-2b": "google/gemma-2b-it",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "gemma-4-e2b": "google/gemma-4-E2B-it",  # fallback for the re-base experiment
}
VOL = os.environ.get("CH10_VOL", "/vol")          # Modal volume mount point
DATA = f"{VOL}/data"
RUNS = f"{VOL}/runs"
RESULTS = f"{VOL}/results"
NOT_IN_CONTEXT = "The context does not contain the answer."

# RAG prompt template of Chapter 8 (system + numbered passages + question)
SYSTEM = ("Answer the question using only the passages. Answer with a short phrase. "
          f"If the passages do not contain the answer, reply exactly: {NOT_IN_CONTEXT}")


def rag_messages(question, passages):
    ctx = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Passages:\n{ctx}\n\nQuestion: {question}"}]


NO_SYSTEM_ROLE = {"gemma-2b"}   # chat template raises "System role not supported"


def for_model(key, messages):
    """Gemma-2B has no system role: put the system text at the start of the first user turn (blank line between).
    Other models get the messages unchanged."""
    if key not in NO_SYSTEM_ROLE or not messages or messages[0]["role"] != "system":
        return messages
    sys_text, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
    if rest and rest[0]["role"] == "user":
        rest[0]["content"] = sys_text + "\n\n" + rest[0]["content"]
    else:
        rest.insert(0, {"role": "user", "content": sys_text})
    return rest


STOP_AT_END_OF_TURN = {"gemma-2b"}   # generation_config lists only <eos>; chat turns end with <end_of_turn>


def generation_config_dir(key):
    """For Gemma-2B: a folder with the model's generation_config.json plus <end_of_turn> in eos_token_id, passed to vLLM
    as generation_config= so that generation stops at the end of the assistant turn (vLLM and lm_eval alike).
    Returns None for the other models (their end-of-turn token is already an eos token)."""
    if key not in STOP_AT_END_OF_TURN:
        return None
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    cfg = json.load(open(hf_hub_download(MODELS[key], "generation_config.json")))
    eot = AutoTokenizer.from_pretrained(MODELS[key]).convert_tokens_to_ids("<end_of_turn>")
    eos = cfg.get("eos_token_id", [])
    cfg["eos_token_id"] = sorted(set((eos if isinstance(eos, list) else [eos]) + [eot]))
    d = f"/tmp/{os.environ.get('USER', 'user')}/generation_config/{key}"
    os.makedirs(d, exist_ok=True)
    json.dump(cfg, open(f"{d}/generation_config.json", "w"), indent=1)
    return d


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


class Timer:
    def __enter__(self):
        self.t0 = time.time(); return self

    def __exit__(self, *a):
        self.s = time.time() - self.t0
