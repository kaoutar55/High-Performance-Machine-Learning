"""Chapter 9 case study, step 2: serve every compressed variant with vLLM 0.32 and measure speed and quality.

    modal run chapter09/modal_quant.py --what perf                 # speed and capacity on A100, H100 and T4
    modal run chapter09/modal_quant.py --what quality              # KL vs baseline and lm-eval, on the H100
    modal run chapter09/modal_quant.py --what prefill,long         # prefill-heavy and long-context studies
    modal run chapter09/modal_quant.py --what perf --gpus T4 --models gemma --variants fp16,w4a16_gptq_rag

Run chapter09/quantize.py first: the Gemma variants and some Llama variants are read from the "hpml-ch09-ckpt"
volume. The other Llama variants are Red Hat's published checkpoints.

Studies
  perf     startup facts (weight memory, KV cache tokens, kernels chosen), single-request latency (0.25 req/s),
           a load sweep, and three repeats at the highest passing rate and the next one (capacity, as in Ch 8)
  prefill  4,096-token prompts with 1 output token at increasing rates: TTFT and the prefill rate a GPU sustains
  long     4,096-token prompts with 1,024-token answers: KV cache use, preemptions and goodput (as in Ch 8)
  quality  on the H100: per-token KL divergence of each variant against the 16-bit model on 200 held-out RAG
           prompts, top-1 agreement, identical-answer rate, and lm-evaluation-harness tasks

Results: chapter09/results/<gpu>_<model>.json (perf, prefill, long) and chapter09/results/quality_<model>.json.
"""
import gc
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import time

import modal

HERE = pathlib.Path(__file__).resolve().parent
VLLM_VERSION = os.environ.get("CH09_VLLM", "0.32.0")
LMEVAL_VERSION = "0.4.13"

GPUS = {"A100-80GB": "A100-80GB", "H100": "H100!", "T4": "T4"}
REQUIRED_NAME = {"A100-80GB": "SXM4", "H100": "H100"}       # reject PCIe A100s, as in Ch 8

LLAMA, GEMMA = "meta-llama/Llama-3.1-8B-Instruct", "google/gemma-2b-it"
RH = "RedHatAI/Meta-Llama-3.1-8B-Instruct"
# name -> (model, source, extra vLLM flags). Sources starting with "/ckpt/" come from chapter09/quantize.py.
VARIANTS = {
    "llama": {
        "bf16":                (LLAMA, []),
        "fp8_dynamic":         (f"{RH}-FP8-dynamic", []),
        "fp8_online":          (LLAMA, ["--quantization", "fp8_per_tensor"]),
        "w8a8_int8":           (f"{RH}-quantized.w8a8", []),
        "w4a16_gptq":          (f"{RH}-quantized.w4a16", []),
        "w4a16_gptq_marlin":   (f"{RH}-quantized.w4a16", ["--linear-backend", "marlin"]),
        "kv_fp8":              (LLAMA, ["--kv-cache-dtype", "fp8"]),
        # the two FP8-KV variants share weights and activations (same llm-compressor FP8_DYNAMIC pipeline) and
        # differ only in the KV scales: 1.0 (none in the checkpoint) vs calibrated per-tensor scales
        "fp8_dynamic_kv_scale1": ("/ckpt/llama-fp8_dynamic", ["--kv-cache-dtype", "fp8"]),
        "fp8_dynamic_kv_cal":  ("/ckpt/llama-fp8_dynamic_kv", ["--kv-cache-dtype", "fp8"]),
        "w4a16_gptq_rag":      ("/ckpt/llama-w4a16_gptq_rag", []),
        "w4a16_gptq_ultrachat": ("/ckpt/llama-w4a16_gptq_ultrachat", []),
        "w4a16_gptq_wikitext": ("/ckpt/llama-w4a16_gptq_wikitext", []),
        "w4a16_awq_rag":       ("/ckpt/llama-w4a16_awq_rag", []),
        "sparse24_sparsegpt":  ("/ckpt/llama-sparse24_sparsegpt", []),
    },
    "gemma": {
        "fp16":                (GEMMA, []),
        "fp8_dynamic":         ("/ckpt/gemma-fp8_dynamic", []),
        "w8a8_int8":           ("/ckpt/gemma-w8a8_int8", []),
        "w4a16_gptq_rag":      ("/ckpt/gemma-w4a16_gptq_rag", []),
        "w4a16_gptq_wikitext": ("/ckpt/gemma-w4a16_gptq_wikitext", []),
        "w4a16_awq_rag":       ("/ckpt/gemma-w4a16_awq_rag", []),
        "kv_fp8":              (GEMMA, ["--kv-cache-dtype", "fp8"]),
        "sparse24_magnitude":  ("/ckpt/gemma-sparse24_magnitude", []),
        "sparse24_wanda":      ("/ckpt/gemma-sparse24_wanda", []),
        "sparse24_sparsegpt":  ("/ckpt/gemma-sparse24_sparsegpt", []),
    },
}
BASELINE = {"llama": "bf16", "gemma": "fp16"}

# Which variants each study runs, per GPU. T4 has no BF16 and 16 GB, so Llama runs there only quantized.
PERF = {
    ("H100", "llama"): ["bf16", "fp8_dynamic", "fp8_online", "w8a8_int8", "w4a16_gptq", "w4a16_gptq_marlin", "kv_fp8"],
    ("A100-80GB", "llama"): ["bf16", "fp8_dynamic", "w8a8_int8", "w4a16_gptq", "kv_fp8"],
    ("T4", "llama"): ["w4a16_gptq", "w8a8_int8", "fp8_online"],
    ("H100", "gemma"): ["fp16", "fp8_dynamic", "w4a16_gptq_rag"],
    ("T4", "gemma"): ["fp16", "w8a8_int8", "w4a16_gptq_rag", "fp8_dynamic", "kv_fp8"],
}
PREFILL = {("H100", "llama"): ["bf16", "fp8_dynamic", "w8a8_int8", "w4a16_gptq"],
           ("A100-80GB", "llama"): ["bf16", "fp8_dynamic", "w8a8_int8", "w4a16_gptq"]}
LONG = {("H100", "llama"): ["bf16", "kv_fp8", "fp8_dynamic_kv_scale1", "fp8_dynamic_kv_cal", "w4a16_gptq"],
        ("A100-80GB", "llama"): ["bf16", "kv_fp8", "fp8_dynamic_kv_scale1", "fp8_dynamic_kv_cal", "w4a16_gptq"]}
QUALITY = {"llama": [v for v in VARIANTS["llama"] if v != "w4a16_gptq_marlin"],
           "gemma": list(VARIANTS["gemma"])}

RATES = {("H100", "llama"): [16, 32, 48, 64, 80, 96], ("A100-80GB", "llama"): [4, 8, 12, 16, 20, 24],
         ("T4", "llama"): [0.25, 0.5, 1, 2, 3], ("H100", "gemma"): [32, 64, 96, 128, 160],
         ("T4", "gemma"): [0.5, 1, 2, 3, 4, 6]}
LONG_RATES = {"H100": [1, 2, 3, 4, 6], "A100-80GB": [0.5, 1, 1.5, 2, 3]}
PREFILL_RATES = [0.5, 1, 2, 4, 6, 8, 12, 16]
SLO = {"ttft_ms": 500, "tpot_ms": 50}
SLO_LONG = {"ttft_ms": 2000, "tpot_ms": 50}
REPEATS = 3

hf_cache = modal.Volume.from_name("hpml-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("hpml-ch08-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("hpml-ch09-ckpt", create_if_missing=True)
vllm_cache = modal.Volume.from_name("hpml-ch09-vllm-cache", create_if_missing=True)

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install(f"vllm=={VLLM_VERSION}", "aiohttp", f"lm-eval[api]=={LMEVAL_VERSION}", "numpy")
         .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "VLLM_CACHE_ROOT": "/vllm_cache",
               "VLLM_USE_FLASHINFER_SAMPLER": "0"})                      # same reason as in Ch 8
         .add_local_file(str(HERE.parent / "chapter08" / "loadgen.py"), "/root/ch09/loadgen.py"))
app = modal.App("hpml-ch09-quant")
COMMON = dict(image=image, timeout=8 * 3600, memory=65536,
              volumes={"/cache": hf_cache, "/data": data_vol, "/ckpt": ckpt_vol, "/vllm_cache": vllm_cache},
              secrets=[modal.Secret.from_name("huggingface")])

SINGLE, LONGP = "/data/prompts_single.jsonl", "/data/prompts_long.jsonl"
EVAL_ROWS = (800, 1000)                                   # held out from calibration (rows 0-511)


def dtype_for(gpu, model):
    return "float16" if model == "gemma" or gpu == "T4" else "bfloat16"


# ------------------------------------------------------------------------------------------------ server
class Server:
    """`vllm serve` for one variant; records the startup facts that explain the results."""

    def __init__(self, gpu, model, variant, max_len=2048):
        src, extra = VARIANTS[model][variant]
        self.cmd = ["vllm", "serve", src, "--dtype", dtype_for(gpu, model), "--max-model-len", str(max_len),
                    "--port", "8000", "--no-enable-prefix-caching", "--served-model-name", "m"] + extra
        self.log_path = "/tmp/vllm.log"

    def __enter__(self):
        import urllib.request
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(1500):
            if self.proc.poll() is not None:
                raise RuntimeError("vllm exited:\n" + open(self.log_path).read()[-12000:])
            try:
                urllib.request.urlopen("http://localhost:8000/health", timeout=2)
                return self
            except Exception:
                time.sleep(1)
        self.__exit__(None, None, None)
        raise TimeoutError("vllm did not become healthy in 25 minutes")

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()
        time.sleep(5)

    def startup(self):
        text = open(self.log_path).read()
        def num(pat, cast=float):
            m = re.search(pat, text)
            return cast(m.group(1).replace(",", "")) if m else None
        keys = ("Selected", "Using", "Marlin", "Machete", "Humming", "Cutlass", "CUTLASS", "attention backend",
                "Falling back", "kv cache", "KV cache", "k_scale", "scaling factor", "quantiz", "does not have native")
        return {"weights_gib": num(r"Model loading took ([\d.]+) GiB"),
                "load_s": num(r"Model loading took [\d.]+ GiB memory and ([\d.]+) seconds"),
                "kv_cache_gib": num(r"Available KV cache memory: ([\d.]+) GiB"),
                "kv_cache_tokens": num(r"KV cache size: ([\d,]+) tokens", int),
                "lines": [l.strip()[-260:] for l in text.splitlines() if any(k in l for k in keys)
                          and "Capturing" not in l][:40]}

    @staticmethod
    def metrics(prefixes=("vllm:",)):
        import urllib.request
        out = {}
        for line in urllib.request.urlopen("http://localhost:8000/metrics", timeout=10).read().decode().splitlines():
            if line.startswith(prefixes) and not line.startswith("#"):
                name, _, val = line.rpartition(" ")
                try:
                    out[name] = float(val)
                except ValueError:
                    pass
        return out

    def load(self, prompts, rate, num, offset=0, seed=0, slo=SLO, max_tokens=128):
        r = subprocess.run([sys.executable, "/root/ch09/loadgen.py", "--model", "m", "--prompts", prompts,
                            "--rate", str(rate), "--num", str(num), "--max-tokens", str(max_tokens),
                            "--slo-ttft-ms", str(slo["ttft_ms"]), "--slo-tpot-ms", str(slo["tpot_ms"]),
                            "--offset", str(offset), "--seed", str(seed)], capture_output=True, text=True)
        return {"error": r.stderr[-2000:]} if r.returncode else json.loads(r.stdout)


def num_for(rate):
    return int(min(3000, max(60, rate * 30)))


def gpu_check(gpu):
    name = subprocess.getoutput("nvidia-smi --query-gpu=name --format=csv,noheader")
    return None if REQUIRED_NAME.get(gpu, "") in name else name


def env_info():
    return {"gpu": subprocess.getoutput("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
            "vllm": subprocess.getoutput(f"{sys.executable} -c 'import vllm; print(vllm.__version__)'")}


# ------------------------------------------------------------------------------------------------ perf
def study_perf(gpu, model, variant):
    res = {}
    with Server(gpu, model, variant) as s:                 # startup facts and the single-request floor
        res["startup"] = s.startup()
        s.load(SINGLE, 0.5, 10, offset=900)
        res["single"] = s.load(SINGLE, 0.25, 30, offset=0)
    rates, sweep = RATES[(gpu, model)], []
    with Server(gpu, model, variant) as s:                 # one pass up the rates, stopping past saturation
        s.load(SINGLE, rates[0], num_for(rates[0]), offset=850)
        for i, rate in enumerate(rates):
            r = s.load(SINGLE, rate, num_for(rate), offset=(i * 131) % 800, seed=i)
            sweep.append(r)
            if r.get("slo_attainment", 0) < 0.5:
                break
    res["sweep"] = sweep
    passing = [r["rate"] for r in sweep if r.get("slo_attainment", 0) >= 0.9]
    top = max(passing) if passing else rates[0]
    nxt = [x for x in rates if x > top][:1]
    knee_rates = [top] + nxt
    repeats = []
    for rep in range(REPEATS):                             # capacity: every repeat on a fresh server
        order = knee_rates[rep % len(knee_rates):] + knee_rates[:rep % len(knee_rates)]
        runs = {}
        with Server(gpu, model, variant) as s:
            s.load(SINGLE, rates[0], int(rates[0] * 60), offset=850, seed=900 + rep)
            for i, rate in enumerate(order):
                r = s.load(SINGLE, rate, num_for(rate), offset=(rep * 211 + i * 131) % 800, seed=200 + 10 * rep + i)
                r["position_in_repeat"] = i
                runs[str(rate)] = r
        repeats.append({"order": order, "runs": runs})
    res["knee"] = {"rates": knee_rates, "repeats": repeats}
    return res


def study_prefill(gpu, model, variant):
    out = []
    with Server(gpu, model, variant, max_len=6144) as s:
        s.load(LONGP, 0.5, 10, offset=900, slo=SLO_LONG, max_tokens=1)
        for i, rate in enumerate(PREFILL_RATES):
            r = s.load(LONGP, rate, int(min(600, max(40, rate * 30))), offset=(i * 61) % 800, seed=i,
                       slo=SLO_LONG, max_tokens=1)
            out.append(r)
            if r.get("ttft_ms", {}).get("p50", 0) > 10000 or r.get("request_throughput", rate) < 0.7 * rate:
                break
        startup = s.startup()
    return {"startup": startup, "runs": out}


def study_long(gpu, model, variant):
    out = []
    keys = ("vllm:kv_cache_usage_perc", "vllm:num_preemptions", "vllm:num_requests_running", "vllm:num_requests_waiting")
    with Server(gpu, model, variant, max_len=6144) as s:
        startup = s.startup()
        s.load(LONGP, 0.5, 10, offset=900, slo=SLO_LONG, max_tokens=64)
        for i, rate in enumerate(LONG_RATES[gpu]):
            before, peak, stop = s.metrics(keys), {}, threading.Event()

            def sample():
                while not stop.is_set():
                    try:
                        for k, v in s.metrics(keys).items():
                            peak[k] = max(v, peak.get(k, v))
                    except Exception:
                        pass
                    stop.wait(0.5)
            th = threading.Thread(target=sample, daemon=True); th.start()
            r = s.load(LONGP, rate, int(min(300, max(40, rate * 60))), offset=(i * 61) % 800, seed=i,
                       slo=SLO_LONG, max_tokens=1024)
            stop.set(); th.join()
            r["metrics_before"], r["metrics_after"], r["metrics_peak"] = before, s.metrics(keys), peak
            out.append(r)
    return {"startup": startup, "runs": out}


STUDIES = {"perf": study_perf, "prefill": study_prefill, "long": study_long}


def run_variant(gpu: str, model: str, variant: str, study: str) -> dict:
    bad = gpu_check(gpu)
    if bad:
        return {"error": f"wrong GPU variant: {bad}", "retry": True}
    t0 = time.time()
    try:
        out = STUDIES[study](gpu, model, variant)
    except Exception as e:
        out = {"error": f"{type(e).__name__}: {e}"[-12000:]}
    out["env"], out["wall_s"] = env_info(), round(time.time() - t0, 1)
    return out


FUNCS = {g: app.function(name=f"bench_{g.lower().replace('-', '_')}", gpu=GPUS[g], cpu=8, **COMMON)(run_variant)
         for g in GPUS}


# ------------------------------------------------------------------------------------------------ quality
K = 20


def eval_prompts():
    rows = [json.loads(l)["prompt"] for l in open(SINGLE)]
    return rows[EVAL_ROWS[0]:EVAL_ROWS[1]]


def _norm(t):
    """SQuAD answer normalization: lower case, no punctuation, no articles, single spaces."""
    import string
    t = "".join(ch for ch in t.lower() if ch not in set(string.punctuation))
    return " ".join(w for w in t.split() if w not in ("a", "an", "the"))


def gold_answers(prompts):
    """SQuAD reference answers for each RAG prompt, matched through the question text of the prompt."""
    from datasets import load_dataset
    gold = {}
    for ex in load_dataset("rajpurkar/squad", split="validation"):
        gold.setdefault(ex["question"], set()).update(ex["answers"]["text"])
    out = []
    for p in prompts:
        q = p.rsplit("Question: ", 1)[-1].split("\nAnswer:")[0].strip()
        out.append(sorted(gold.get(q, [])))
    return out


def answer_scores(texts, golds):
    """Per prompt: does the first line of the answer contain a reference answer, and the best token F1."""
    import collections
    contains, f1 = [], []
    for t, g in zip(texts, golds):
        if not g:
            contains.append(None); f1.append(None); continue
        a = _norm(t.strip().split("\n")[0])
        contains.append(float(any(_norm(x) and _norm(x) in a for x in g)))
        best = 0.0
        for x in g:
            pa, pg = a.split(), _norm(x).split()
            common = sum((collections.Counter(pa) & collections.Counter(pg)).values())
            if common:
                pr, rc = common / len(pa), common / len(pg)
                best = max(best, 2 * pr * rc / (pr + rc))
        f1.append(best)
    return contains, f1


def paired(base, var):
    """Mean of each, the paired difference with its standard error, and the flips, over items scored in both."""
    import numpy as np
    pairs = [(b, v) for b, v in zip(base, var) if b is not None and v is not None]
    b, v = np.array([x[0] for x in pairs]), np.array([x[1] for x in pairs])
    d = v - b
    return {"n": int(len(d)), "baseline": float(b.mean()), "variant": float(v.mean()), "diff": float(d.mean()),
            "diff_se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None,
            "flips_to_wrong": int(((b == 1) & (v == 0)).sum()), "flips_to_right": int(((b == 0) & (v == 1)).sum())}


def make_llm(model, variant, max_logprobs=K):
    from vllm import LLM
    src, extra = VARIANTS[model][variant]
    kw = dict(model=src, dtype=dtype_for("H100", model), max_model_len=2048, max_logprobs=max_logprobs,
              enable_prefix_caching=False, gpu_memory_utilization=0.85, seed=0)
    if "--quantization" in extra:
        kw["quantization"] = extra[extra.index("--quantization") + 1]
    if "--kv-cache-dtype" in extra:
        kw["kv_cache_dtype"] = extra[extra.index("--kv-cache-dtype") + 1]
    return LLM(**kw)


def free(llm):
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def topk_from(entry):
    """vLLM prompt-logprob dict -> (ids, logps) of the K highest-ranked tokens."""
    items = sorted(entry.items(), key=lambda kv: kv[1].rank)[:K]
    return [t for t, _ in items], [lp.logprob for _, lp in items]


def score_ids(llm, p_ids, r_ids, ids):
    """Log-probs the model assigns to the reference's top-K token ids at every answer position."""
    import numpy as np
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    try:
        q = []
        for p, r, I in zip(p_ids, r_ids, ids):
            sp = SamplingParams(max_tokens=1, prompt_logprob_token_ids=np.asarray(I, dtype=np.int32),
                                prompt_logprob_start=len(p) - 1)
            o = llm.generate(TokensPrompt(prompt_token_ids=list(p) + list(r)), sp, use_tqdm=False)[0]
            q.append(np.asarray(o.prompt_token_id_logprobs, float)[:len(r)])
        return q, "prompt_logprob_token_ids"
    except Exception as e:                                 # fall back to the top-K intersection
        method = f"top-k intersection (token-id scoring failed: {type(e).__name__}: {str(e)[:200]})"
    q = []
    scored = llm.generate([TokensPrompt(prompt_token_ids=list(p) + list(r)) for p, r in zip(p_ids, r_ids)],
                          SamplingParams(max_tokens=1, prompt_logprobs=K))
    for p, r, I, o in zip(p_ids, r_ids, ids, scored):
        rows = []
        for t in range(len(r)):
            d = o.prompt_logprobs[len(p) + t]
            floor = min(lp.logprob for lp in d.values())
            rows.append([d[i].logprob if i in d else floor for i in I[t]])
        q.append(np.array(rows))
    return q, method


@app.function(gpu="H100!", cpu=8, **COMMON)
def reference(model: str, refresh: bool = False) -> dict:
    """16-bit model: greedy answers for the held-out prompts and the top-K next-token distribution at every
    answer position. Saved to the checkpoint volume so every variant is compared with the same reference."""
    import numpy as np
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    path = f"/ckpt/reference_{model}.npz"
    if os.path.exists(path) and not refresh:
        return json.load(open(path.replace(".npz", ".json")))
    llm = make_llm(model, BASELINE[model])
    tok = llm.get_tokenizer()
    prompts = eval_prompts()
    p_ids = [tok(p, add_special_tokens=True)["input_ids"] for p in prompts]
    gen = llm.generate([TokensPrompt(prompt_token_ids=p) for p in p_ids],
                       SamplingParams(temperature=0.0, max_tokens=128))
    r_ids = [list(o.outputs[0].token_ids) for o in gen]
    scored = llm.generate([TokensPrompt(prompt_token_ids=p + r) for p, r in zip(p_ids, r_ids)],
                          SamplingParams(max_tokens=1, prompt_logprobs=K))
    ids, lps = [], []
    for p, r, o in zip(p_ids, r_ids, scored):
        rows = [topk_from(o.prompt_logprobs[len(p) + t]) for t in range(len(r))]
        ids.append(np.array([x[0] for x in rows], dtype=np.int32)); lps.append(np.array([x[1] for x in rows]))
    # noise floor: the same model scored with the same method the variants use (one prompt per call, so
    # the batch composition differs from the reference pass); this also checks the scoring path itself
    noise, method = score_ids(llm, p_ids, r_ids, ids)
    np.savez(path, p_ids=np.array(p_ids, dtype=object), r_ids=np.array(r_ids, dtype=object),
             ids=np.array(ids, dtype=object), lps=np.array(lps, dtype=object),
             texts=np.array([o.outputs[0].text for o in gen], dtype=object))
    nf = kl_rows(lps, noise); nf["method"] = method
    info = {"model": model, "prompts": len(p_ids), "answer_tokens": int(sum(len(r) for r in r_ids)),
            "noise_floor": nf, "env": env_info()}
    json.dump(info, open(path.replace(".npz", ".json"), "w"), indent=2)
    ckpt_vol.commit()
    free(llm)
    return info


def kl_rows(ref_lps, q_lps):
    """Top-K KL with a 'rest' bucket: a lower bound on the true per-token KL (data-processing inequality)."""
    import numpy as np
    kls, top1 = [], []
    for p_l, q_l in zip(ref_lps, q_lps):
        p, q = np.exp(np.asarray(p_l, float)), np.exp(np.asarray(q_l, float))
        pr, qr = np.clip(1 - p.sum(1), 1e-12, 1), np.clip(1 - q.sum(1), 1e-12, 1)
        kl = (p * (np.asarray(p_l) - np.asarray(q_l))).sum(1) + pr * (np.log(pr) - np.log(qr))
        kls.append(kl); top1.append(np.argmax(q_l, 1) == 0)
    kl = np.concatenate(kls); t1 = np.concatenate(top1)
    return {"kl_mean": float(kl.mean()), "kl_p50": float(np.median(kl)), "kl_p99": float(np.percentile(kl, 99)),
            "kl_max": float(kl.max()), "top1_agreement": float(t1.mean()), "tokens": int(kl.size)}


def lm_eval(model, variant, out_dir):
    src, extra = VARIANTS[model][variant]
    args = f"pretrained={src},dtype={dtype_for('H100', model)},max_model_len=4096,gpu_memory_utilization=0.85,seed=0"
    if "--quantization" in extra:
        args += f",quantization={extra[extra.index('--quantization') + 1]}"
    if "--kv-cache-dtype" in extra:
        args += f",kv_cache_dtype={extra[extra.index('--kv-cache-dtype') + 1]}"
    gen = {"llama": "gsm8k_cot_llama,ifeval,arc_challenge_llama", "gemma": "ifeval"}[model]
    ll = {"llama": "arc_challenge,wikitext", "gemma": "arc_challenge,wikitext"}[model]
    runs = {}
    for tag, tasks, flags in (("gen", gen, ["--apply_chat_template", "--fewshot_as_multiturn"]),
                              ("ll", ll, [])):
        a = args + (",add_bos_token=True" if tag == "ll" else "")
        cmd = ["lm_eval", "--model", "vllm", "--model_args", a, "--tasks", tasks, "--batch_size", "auto",
               "--seed", "1234", "--output_path", f"{out_dir}/{tag}", "--log_samples"] + flags
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        runs[tag] = {"cmd": " ".join(cmd), "returncode": r.returncode, "wall_s": round(time.time() - t0, 1),
                     "stderr_tail": r.stderr[-3000:] if r.returncode else ""}
    metrics, per_item = {}, {}
    for f in pathlib.Path(out_dir).rglob("results_*.json"):
        for task, m in json.load(open(f)).get("results", {}).items():
            metrics[task] = {k: v for k, v in m.items() if isinstance(v, (int, float))}
    keymetric = {"gsm8k_cot_llama": "exact_match", "ifeval": "prompt_level_strict_acc",
                 "arc_challenge_llama": "exact_match", "arc_challenge": "acc"}
    for f in pathlib.Path(out_dir).rglob("samples_*.jsonl"):
        task = next((t for t in keymetric if f.name.startswith(f"samples_{t}_")), None)
        if not task:
            continue
        vals = {}
        for line in open(f):
            s = json.loads(line)
            v = s.get(keymetric[task])
            if v is None:                                  # e.g. per-filter names such as "exact_match,strict-match"
                v = next((s[k] for k in s if k.startswith(keymetric[task])), None)
            if isinstance(v, list):
                v = v[0]
            if v is not None:
                vals[int(s["doc_id"])] = float(v)
        per_item[task] = vals
    return {"runs": runs, "metrics": metrics, "per_item": per_item}


@app.function(gpu="H100!", cpu=8, **COMMON)
def quality(model: str, variant: str) -> dict:
    import numpy as np
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    t0 = time.time()
    out = {"model": model, "variant": variant, "env": env_info()}
    try:
        out["lm_eval"] = lm_eval(model, variant, f"/tmp/lmeval_{model}_{variant}")
    except Exception as e:
        out["lm_eval"] = {"error": f"{type(e).__name__}: {e}"[-4000:]}
    ref = np.load(f"/ckpt/reference_{model}.npz", allow_pickle=True)
    p_ids, r_ids, ids, lps = list(ref["p_ids"]), list(ref["r_ids"]), list(ref["ids"]), list(ref["lps"])
    llm = make_llm(model, variant)
    # 1) same answers? greedy generation compared with the reference text
    gen = llm.generate([TokensPrompt(prompt_token_ids=list(p)) for p in p_ids],
                       SamplingParams(temperature=0.0, max_tokens=128))
    same, prefix = [], []
    for o, r in zip(gen, r_ids):
        g = list(o.outputs[0].token_ids); r = list(r)
        same.append(g == r)
        n = 0
        while n < min(len(g), len(r)) and g[n] == r[n]:
            n += 1
        prefix.append(n)
    out["generation"] = {"identical_answers": float(np.mean(same)), "mean_common_prefix_tokens": float(np.mean(prefix)),
                         "answers": len(same)}
    # 1b) correctness against the SQuAD reference answers, paired with the 16-bit model's answers
    try:
        golds = gold_answers(eval_prompts())
        b_c, b_f = answer_scores(list(ref["texts"]), golds)
        v_c, v_f = answer_scores([o.outputs[0].text for o in gen], golds)
        out["squad"] = {"prompts_with_reference": int(sum(1 for g in golds if g)),
                        "contains_answer": paired(b_c, v_c), "token_f1": paired(b_f, v_f),
                        "per_item": {"baseline_contains": b_c, "variant_contains": v_c}}
    except Exception as e:
        out["squad"] = {"error": f"{type(e).__name__}: {e}"[-2000:]}
    # 2) per-token KL on the reference answers, scoring exactly the reference's top-K token ids
    q_lps, method = score_ids(llm, p_ids, r_ids, ids)
    out["kl"] = kl_rows(lps, q_lps)
    out["kl"]["method"] = method
    free(llm)
    out["wall_s"] = round(time.time() - t0, 1)
    return out


@app.function(gpu="H100!", cpu=8, **COMMON)
def squad(model: str, variant: str) -> dict:
    """SQuAD correctness only (greedy answers on the held-out prompts), for quality results produced before
    this metric was added to quality(). Much cheaper than a full quality run: no lm-eval and no KL."""
    import numpy as np
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    ref = np.load(f"/ckpt/reference_{model}.npz", allow_pickle=True)
    llm = make_llm(model, variant)
    gen = llm.generate([TokensPrompt(prompt_token_ids=list(p)) for p in ref["p_ids"]],
                       SamplingParams(temperature=0.0, max_tokens=128))
    free(llm)
    golds = gold_answers(eval_prompts())
    b_c, b_f = answer_scores(list(ref["texts"]), golds)
    v_c, v_f = answer_scores([o.outputs[0].text for o in gen], golds)
    return {"prompts_with_reference": int(sum(1 for g in golds if g)),
            "contains_answer": paired(b_c, v_c), "token_f1": paired(b_f, v_f),
            "per_item": {"baseline_contains": b_c, "variant_contains": v_c}}


# ------------------------------------------------------------------------------------------------ entrypoint
def merge(path, key, value):
    data = json.loads(path.read_text()) if path.exists() else {}
    data.setdefault(key[0], {})[key[1]] = value
    path.write_text(json.dumps(data, indent=2))


@app.local_entrypoint()
def main(what: str = "perf", gpus: str = "", models: str = "", variants: str = ""):
    what = [w for w in what.split(",") if w]
    sel_g = [g for g in (gpus.split(",") if gpus else GPUS)]
    sel_m = [m for m in (models.split(",") if models else VARIANTS)]
    sel_v = set(v for v in variants.split(",") if v)
    out = HERE / "results"; out.mkdir(exist_ok=True)
    if "quality" in what:
        for m in sel_m:                                    # the reference first: every variant is compared with it
            r = reference.remote(m)                        # cached on the volume after the first run
            (out / f"reference_{m}.json").write_text(json.dumps(r, indent=2))
            print("reference", m, r.get("noise_floor"))
        jobs = [(m, v) for m in sel_m for v in QUALITY[m] if not sel_v or v in sel_v]
        for (m, v), r in zip(jobs, quality.starmap(jobs, return_exceptions=True)):
            r = r if isinstance(r, dict) else {"error": repr(r)[-4000:]}
            merge(out / f"quality_{m}.json", ("variants", v), r)
            print("quality", m, v, r.get("kl", r.get("error", "")))
    if "squad" in what:                                    # add SQuAD scores to existing quality results
        jobs = [(m, v) for m in sel_m for v in QUALITY[m] if not sel_v or v in sel_v]
        for (m, v), r in zip(jobs, squad.starmap(jobs, return_exceptions=True)):
            r = r if isinstance(r, dict) else {"error": repr(r)[-2000:]}
            path = out / f"quality_{m}.json"
            data = json.loads(path.read_text()) if path.exists() else {}
            data.setdefault("variants", {}).setdefault(v, {})["squad"] = r
            path.write_text(json.dumps(data, indent=2))
            print("squad", m, v, r.get("contains_answer", r.get("error", "")))
    table = {"perf": PERF, "prefill": PREFILL, "long": LONG}
    for study in [w for w in what if w in table]:
        jobs = [(g, m, v, study) for (g, m), vs in table[study].items() if g in sel_g and m in sel_m
                for v in vs if not sel_v or v in sel_v]
        handles = [(j, FUNCS[j[0]].spawn(*j)) for j in jobs]
        for j, h in handles:
            for attempt in range(6):
                try:
                    r = h.get()
                except Exception as e:
                    r = {"error": f"{type(e).__name__}: {e}"[-4000:]}
                if not r.get("retry"):
                    break
                print(j, r["error"], "retrying")
                h = FUNCS[j[0]].spawn(*j)
            g, m, v, s = j
            path = out / f"{g}_{m}.json"
            if "error" in r and not any(k in r for k in ("startup", "runs", "single")):
                merge(out / f"{g}_{m}.errors.json", (s, v), r)   # never overwrite a good result with an error
                print("FAILED", j, r["error"][:300])
            else:
                merge(path, (s, v), r)
                print("done", j)
