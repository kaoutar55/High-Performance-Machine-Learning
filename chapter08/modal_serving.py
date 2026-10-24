"""Chapter 8 case study on Modal: serve the RAG generator with vLLM and measure it under load.

Two generators:
  gemma   google/gemma-2b-it in FP16 (the book's case-study generator; one KV head, 18 KB of KV per token)
          on the A100 80GB, the H100 and the T4
  llama   meta-llama/Llama-3.1-8B-Instruct in BF16 (8 KV heads, 128 KB of KV per token) on the A100 and the H100

    modal run chapter08/modal_serving.py                                  # every GPU and model, every study
    modal run chapter08/modal_serving.py --gpus T4 --models gemma
    modal run chapter08/modal_serving.py --studies single,sweep

Studies (each runs on a freshly started `vllm serve`):
  single   one request at a time (0.25 req/s): the latency floor, to compare with the eager and compiled baselines
  sweep    default configuration, Poisson arrivals at increasing rates: throughput, latency percentiles, goodput
  long     4,096-token RAG prompts, 1,024-token answers, at increasing rates: KV cache usage, preemptions, and
           whether prefill compute or KV capacity caps the load
  prefix   RAG follow-up turns with automatic prefix caching on and off: TTFT of the second turn
  budget   max-num-batched-tokens 512 vs 8192 at a high rate: the chunked-prefill trade-off (TTFT vs ITL p99)
  spec     n-gram (prompt-lookup) speculative decoding vs none, at a low and a high rate, plus acceptance metrics

Results go to chapter08/results/<gpu>_<model>.json. The Hugging Face token comes from the Modal secret
"huggingface" (it needs access to both gated models); it is never written to the results.
"""
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

import modal

HERE = pathlib.Path(__file__).resolve().parent
VLLM_VERSION = os.environ.get("CH08_VLLM", "0.32.0")
GPUS = {"A100-80GB": "A100-80GB", "H100": "H100!", "T4": "T4"}
MODELS = {
    "gemma": {"id": "google/gemma-2b-it", "dtype": "float16", "gpus": ["A100-80GB", "H100", "T4"],
              "rates": {"A100-80GB": [1, 2, 4, 8, 16, 32, 48], "H100": [1, 2, 4, 8, 16, 32, 48, 64],
                        "T4": [0.5, 1, 2, 4, 8, 12]},
              "long_rates": {"A100-80GB": [0.5, 1, 2, 4, 8], "H100": [0.5, 1, 2, 4, 8, 12], "T4": [0.25, 0.5, 1]},
              "studies": ["single", "sweep", "long", "prefix", "budget", "spec"]},
    "llama": {"id": "meta-llama/Llama-3.1-8B-Instruct", "dtype": "bfloat16", "gpus": ["A100-80GB", "H100"],
              "rates": {"A100-80GB": [1, 2, 4, 8, 16, 24, 32], "H100": [1, 2, 4, 8, 16, 32, 48]},
              "long_rates": {"A100-80GB": [0.25, 0.5, 1, 2, 3, 4], "H100": [0.25, 0.5, 1, 2, 4, 6]},
              "studies": ["single", "sweep", "long", "prefix", "spec"]},
}
SLO = {"ttft_ms": 500, "tpot_ms": 50}
SLO_LONG = {"ttft_ms": 2000, "tpot_ms": 50}
MAX_TOKENS = 128
MAX_TOKENS_LONG = 1024                                   # long outputs keep each request's KV resident for longer

hf_cache = modal.Volume.from_name("hpml-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("hpml-ch08-data", create_if_missing=True)

prompt_image = (modal.Image.debian_slim(python_version="3.11")
                .pip_install("torch==2.14.1", "transformers==5.20.2", "sentence-transformers==6.2.0",
                             "faiss-cpu", "datasets")
                .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
                .add_local_file(str(HERE / "build_prompts.py"), "/root/ch08/build_prompts.py"))
serve_image = (modal.Image.debian_slim(python_version="3.11")
               .pip_install(f"vllm=={VLLM_VERSION}", "aiohttp")
               .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
               .add_local_file(str(HERE / "loadgen.py"), "/root/ch08/loadgen.py"))

app = modal.App("hpml-ch08-serving")
COMMON = dict(timeout=6 * 3600, volumes={"/cache": hf_cache, "/data": data_vol},
              secrets=[modal.Secret.from_name("huggingface")])

SINGLE = "/data/prompts_single.jsonl"
FOLLOW = "/data/prompts_followup.jsonl"
LONG = "/data/prompts_long.jsonl"


@app.function(image=prompt_image, gpu="T4", **COMMON)
def build_prompts() -> str:
    if not os.path.exists(LONG):
        r = subprocess.run([sys.executable, "/root/ch08/build_prompts.py", "--out-dir", "/data", "--num", "1000"],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-3000:])
        data_vol.commit()
        return r.stdout
    return "prompts already built"


# ---------------------------------------------------------------------------------------------------- helpers
class Server:
    """Start `vllm serve` with extra flags, wait until it is healthy, keep its startup log."""

    def __init__(self, model, extra, max_len=2048):
        m = MODELS[model]
        self.model_id = m["id"]
        self.cmd = ["vllm", "serve", m["id"], "--dtype", m["dtype"], "--max-model-len", str(max_len),
                    "--port", "8000"] + extra
        self.log = open("/tmp/vllm.log", "w")

    def __enter__(self):
        import urllib.request
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(1200):
            if self.proc.poll() is not None:
                raise RuntimeError("vllm exited:\n" + open("/tmp/vllm.log").read()[-4000:])
            try:
                urllib.request.urlopen("http://localhost:8000/health", timeout=2)
                return self
            except Exception:
                time.sleep(1)
        raise TimeoutError("vllm did not become healthy in 20 minutes")

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        time.sleep(5)

    def startup_lines(self):
        keys = ("KV cache", "Maximum concurrency", "attention backend", "Using", "graph", "chunked", "prefix")
        return [l.strip()[-240:] for l in open("/tmp/vllm.log") if any(k in l for k in keys)][:40]

    def warnings(self):
        return sum(1 for l in open("/tmp/vllm.log") if "preempt" in l.lower())

    @staticmethod
    def metrics(prefixes=("vllm:",)):
        import urllib.request
        text = urllib.request.urlopen("http://localhost:8000/metrics", timeout=10).read().decode()
        out = {}
        for line in text.splitlines():
            if line.startswith(prefixes) and not line.startswith("#"):
                name, _, val = line.rpartition(" ")
                try:
                    out[name] = float(val)
                except ValueError:
                    pass
        return out

    def load(self, prompts, rate, num, offset=0, seed=0, slo=SLO, max_tokens=MAX_TOKENS):
        r = subprocess.run([sys.executable, "/root/ch08/loadgen.py", "--model", self.model_id, "--prompts", prompts,
                            "--rate", str(rate), "--num", str(num), "--max-tokens", str(max_tokens),
                            "--slo-ttft-ms", str(slo["ttft_ms"]), "--slo-tpot-ms", str(slo["tpot_ms"]),
                            "--offset", str(offset), "--seed", str(seed)], capture_output=True, text=True)
        if r.returncode:
            return {"error": r.stderr[-2000:]}
        return json.loads(r.stdout)


def num_for(rate):
    return int(min(600, max(60, rate * 40)))          # about 40 s of arrivals, at least 60 requests


# ---------------------------------------------------------------------------------------------------- studies
def study_single(gpu, model):
    with Server(model, []) as s:
        s.load(SINGLE, 0.5, 10, offset=900)                    # warm-up
        return {"startup": s.startup_lines(), "result": s.load(SINGLE, 0.25, 30, offset=0)}


def study_sweep(gpu, model):
    out = []
    with Server(model, []) as s:
        s.load(SINGLE, 2, 20, offset=900)
        for i, rate in enumerate(MODELS[model]["rates"][gpu]):
            out.append(s.load(SINGLE, rate, num_for(rate), offset=(i * 97) % 800, seed=i))
        return {"startup": s.startup_lines(), "runs": out}


def study_long(gpu, model):
    out = []
    keys = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "vllm:num_preemptions", "vllm:num_requests_running",
            "vllm:num_requests_waiting")
    with Server(model, [], max_len=6144) as s:
        s.load(LONG, 0.5, 10, offset=900, slo=SLO_LONG, max_tokens=64)
        for i, rate in enumerate(MODELS[model]["long_rates"][gpu]):
            before = s.metrics(keys)
            peak, stop = {}, threading.Event()

            def sample():                                       # peak KV usage and queue depth during the run
                while not stop.is_set():
                    try:
                        for k, v in s.metrics(keys).items():
                            peak[k] = max(v, peak.get(k, v))
                    except Exception:
                        pass
                    stop.wait(0.5)
            th = threading.Thread(target=sample, daemon=True); th.start()
            r = s.load(LONG, rate, int(min(300, max(40, rate * 60))), offset=(i * 61) % 800, seed=i, slo=SLO_LONG,
                       max_tokens=MAX_TOKENS_LONG)
            stop.set(); th.join()
            r["metrics_before"], r["metrics_after"], r["metrics_peak"] = before, s.metrics(keys), peak
            out.append(r)
        return {"startup": s.startup_lines(), "runs": out, "preemption_log_lines": s.warnings(), "slo": SLO_LONG}


def study_prefix(gpu, model):
    res = {}
    for name, flag in (("prefix_on", ["--enable-prefix-caching"]), ("prefix_off", ["--no-enable-prefix-caching"])):
        with Server(model, flag) as s:
            first = s.load(SINGLE, 4, 300, offset=0)           # first turns fill the cache
            follow = s.load(FOLLOW, 4, 300, offset=0)          # second turns share the first turn's prompt
            res[name] = {"first_turn": first, "follow_up": follow, "metrics": s.metrics(("vllm:prefix_cache",)),
                         "startup": s.startup_lines()}
    return res


def study_budget(gpu, model):
    res = {}
    rate = MODELS[model]["rates"][gpu][-2]
    for budget in (512, 8192):
        with Server(model, ["--max-num-batched-tokens", str(budget)]) as s:
            s.load(SINGLE, 2, 20, offset=900)
            res[f"budget_{budget}"] = {"rate": rate, "result": s.load(SINGLE, rate, num_for(rate), offset=100)}
    return res


def study_spec(gpu, model):
    res = {}
    cfg = json.dumps({"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_min": 2, "prompt_lookup_max": 5})
    for name, flag in (("no_spec", []), ("ngram", ["--speculative-config", cfg])):
        with Server(model, flag) as s:
            s.load(SINGLE, 1, 10, offset=900)
            runs = {}
            for rate in (0.25, MODELS[model]["rates"][gpu][-2]):
                runs[str(rate)] = s.load(SINGLE, rate, 30 if rate < 1 else num_for(rate), offset=200)
            res[name] = {"runs": runs, "metrics": s.metrics(("vllm:spec_decode",)), "startup": s.startup_lines()}
    return res


STUDIES = {"single": study_single, "sweep": study_sweep, "long": study_long, "prefix": study_prefix,
           "budget": study_budget, "spec": study_spec}


def run_studies(gpu: str, model: str, studies: list) -> dict:
    env = {"gpu": subprocess.getoutput("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
           "vllm": subprocess.getoutput(f"{sys.executable} -c 'import vllm; print(vllm.__version__)'"),
           "model": MODELS[model]["id"], "dtype": MODELS[model]["dtype"], "slo": SLO, "slo_long": SLO_LONG,
           "max_tokens": MAX_TOKENS, "max_tokens_long": MAX_TOKENS_LONG}
    out = {"env": env}
    for name in studies:
        t0 = time.time()
        try:
            out[name] = STUDIES[name](gpu, model)
        except Exception as e:                                   # keep the other studies' results
            out[name] = {"error": f"{type(e).__name__}: {e}"[-4000:]}
        out[name + "_wall_s"] = round(time.time() - t0, 1)
    return out


def _tag(s):
    return s.lower().replace("-", "_")


FUNCS = {(g, m): app.function(name=f"serve_{_tag(g)}_{m}", image=serve_image, gpu=GPUS[g], **COMMON)(run_studies)
         for m, spec in MODELS.items() for g in spec["gpus"]}


@app.local_entrypoint()
def main(gpus: str = "", models: str = "", studies: str = ""):
    sel_g = [g.strip() for g in gpus.split(",") if g.strip()] or list(GPUS)
    sel_m = [m.strip() for m in models.split(",") if m.strip()] or list(MODELS)
    st = [s.strip() for s in studies.split(",") if s.strip()]
    print(build_prompts.remote())
    handles = {}
    for (g, m), fn in FUNCS.items():
        if g in sel_g and m in sel_m:
            handles[(g, m)] = fn.spawn(g, m, [s for s in (st or MODELS[m]["studies"]) if s in MODELS[m]["studies"]])
    outdir = HERE / "results"
    outdir.mkdir(exist_ok=True)
    for (g, m), h in handles.items():
        try:
            res = h.get()
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {e}"}
        (outdir / f"{g}_{m}.json").write_text(json.dumps(res, indent=2))
        print(f"wrote results/{g}_{m}.json")
