"""Build the Chapter 8 serving workload from the case-study RAG pipeline.

Same corpus, embedder, index and prompt template as the Chapter 2 lab (2,000 WikiText-2 passages, BGE-M3,
FAISS inner product, top-5, "Context: ... Question: ... Answer:"), but with 1,000 distinct questions so that
the serving benchmark does not send identical prompts (which would make prefix caching look better than it is).

Writes:
  prompts_single.jsonl    one RAG prompt per question
  prompts_followup.jsonl  a second turn for each question: the first turn's prompt, a short answer, and a
                          follow-up question. Sent after the first turns, its long shared prefix is what
                          prefix caching can reuse.

    python build_prompts.py --out-dir data --num 1000
"""
import argparse
import json
import os
import random

import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

TEMPLATE = "Context:\n{context}\n\nQuestion: {query}\nAnswer:"
FOLLOWUPS = ["Can you explain that in more detail?", "What evidence in the context supports this?",
             "Summarize the answer in one sentence.", "Which passage is the most relevant, and why?"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--max-prompt-tokens", type=int, default=512)
    ap.add_argument("--generator", default="google/gemma-2b-it")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    corpus = [t.strip() for t in ds["text"] if 200 < len(t) < 800][:2000]
    questions = list(dict.fromkeys(load_dataset("rajpurkar/squad", split="validation")["question"]))
    random.Random(0).shuffle(questions)
    questions = questions[:args.num]

    embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")
    emb = embedder.encode(corpus, batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)
    q_emb = embedder.encode(questions, batch_size=128, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    _, ids = index.search(q_emb, 5)

    tok = AutoTokenizer.from_pretrained(args.generator)
    rng = random.Random(1)
    n_tok = []
    with open(os.path.join(args.out_dir, "prompts_single.jsonl"), "w") as f1, \
         open(os.path.join(args.out_dir, "prompts_followup.jsonl"), "w") as f2:
        for q, row in zip(questions, ids):
            context = "\n\n".join(corpus[i] for i in row)
            prompt = TEMPLATE.format(context=context, query=q)
            toks = tok(prompt, add_special_tokens=True)["input_ids"]
            if len(toks) > args.max_prompt_tokens:                 # same 512-token cap as the lab
                prompt = tok.decode(toks[-args.max_prompt_tokens:], skip_special_tokens=True)
            n_tok.append(min(len(toks), args.max_prompt_tokens))
            f1.write(json.dumps({"prompt": prompt}) + "\n")
            second = prompt + " " + "(answer omitted)" + "\nQuestion: " + rng.choice(FOLLOWUPS) + "\nAnswer:"
            f2.write(json.dumps({"prompt": second}) + "\n")
    print(json.dumps({"prompts": len(n_tok), "mean_prompt_tokens": sum(n_tok) / len(n_tok),
                      "max_prompt_tokens": max(n_tok)}))


if __name__ == "__main__":
    main()
