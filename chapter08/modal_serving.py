"""Chapter 8 case study on Modal: serve the RAG generator (Gemma-2B-IT, FP16) with vLLM and measure it under load.

    modal run chapter08/modal_serving.py                       # all three GPUs, all studies
    modal run chapter08/modal_serving.py --gpus T4             # a subset, comma-separated
    modal run chapter08/modal_serving.py --studies sweep,prefix

Studies (each runs on a freshly started `vllm serve`):
  single   one request at a time (0.25 req/s): the latency floor, to compare with the eager baseline of Chapters 2-5
  sweep    default configuration, Poisson arrivals at increasing rates: throughput, latency percentiles, goodput
  prefix   RAG follow-up turns with automatic prefix caching on and off: TTFT of the second turn
  budget   max-num-batched-tokens 512 vs 8192 at a high rate: the chunked-prefill trade-off (TTFT vs ITL p99)
  spec     n-gram (prompt-lookup) speculative decoding vs none, at a low and a high rate, plus acceptance metrics

Results go to chapter08/results/<gpu>.json. The Hugging Face token comes from the Modal secret "huggingface";
it is never written to the results.
"""
import json
import os
import pathlib
import subprocess
import sys
import time

import modal

HERE = pathlib.Path(__file__).resolve().parent
MODEL = "google/gemma-2b-it"
VLLM_VERSION = os.environ.get("CH08_VLLM", "0.32.0")
GPUS = {"A100-80GB": "A100-80GB", "H100": "H100!", "T4": "T4"}
RATES = {"A100-80GB": [1, 2, 4, 8, 16, 32, 48], "H100": [1, 2, 4, 8, 16, 32, 48, 64], "T4": [0.5, 1, 2, 4, 8, 12]}
SLO = {"ttft_ms": 500, "tpot_ms": 50}
MAX_TOKENS = 128
BASE_ARGS = ["--dtype", "float16", "--max-model-len", "2048", "--port", "8000"]

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
COMMON = dict(timeout=4 * 3600, volumes={"/cache": hf_cache, "/data": data_vol},
              secrets=[modal.Secret.from_name("huggingface")])


@app.function(image=prompt_image, gpu="T4", **COMMON)
def build_prompts() -> str:
    if not os.path.exists("/data/prompts_followup.jsonl"):
        r = subprocess.run([sys.executable, "/root/ch08/build_prompts.py", "--out-dir", "/data", "--num", "1000",
                            "--generator", MODEL], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-3000:])
        data_vol.commit()
        return r.stdout
    return "prompts already built"


# ---------------------------------------------------------------------------------------------------- helpers
class Server:
    """Start `vllm serve` with extra flags, wait until it is healthy, keep its startup log."""

    def __init__(self, extra):
        self.cmd = ["vllm", "serve", MODEL] + BASE_ARGS + extra
        self.log = open("/tmp/vllm.log", "w")

    def __enter__(self):
        import urllib.request
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(900):
            if self.proc.poll() is not None:
                raise RuntimeError("vllm exited:\n" + open("/tmp/vllm.log").read()[-4000:])
            try:
                urllib.request.urlopen("http://localhost:8000/health", timeout=2)
                return self
            except Exception:
                time.sleep(1)
        raise TimeoutError("vllm did not become healthy in 15 minutes")

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

    @staticmethod
    def metrics(prefix="vllm:"):
        import urllib.request
        text = urllib.request.urlopen("http://localhost:8000/metrics", timeout=10).read().decode()
        out = {}
        for line in text.splitlines():
            if line.startswith(prefix) and not line.startswith("#"):
                name, _, val = line.rpartition(" ")
                try:
                    out[name] = float(val)
                except ValueError:
                    pass
        return out


def load(prompts, rate, num, offset=0, seed=0):
    r = subprocess.run([sys.executable, "/root/ch08/loadgen.py", "--model", MODEL, "--prompts", prompts,
                        "--rate", str(rate), "--num", str(num), "--max-tokens", str(MAX_TOKENS),
                        "--slo-ttft-ms", str(SLO["ttft_ms"]), "--slo-tpot-ms", str(SLO["tpot_ms"]),
                        "--offset", str(offset), "--seed", str(seed)], capture_output=True, text=True)
    if r.returncode:
        return {"error": r.stderr[-2000:]}
    return json.loads(r.stdout)


def num_for(rate):
    return int(min(600, max(60, rate * 40)))          # about 40 s of arrivals, at least 60 requests


SINGLE = "/data/prompts_single.jsonl"
FOLLOW = "/data/prompts_followup.jsonl"


# ---------------------------------------------------------------------------------------------------- studies
def study_single(gpu):
    with Server([]) as s:
        load(SINGLE, 0.5, 10, offset=900)                       # warm-up
        return {"startup": s.startup_lines(), "result": load(SINGLE, 0.25, 30, offset=0)}


def study_sweep(gpu):
    out = []
    with Server([]) as s:
        load(SINGLE, 2, 20, offset=900)
        for i, rate in enumerate(RATES[gpu]):
            out.append(load(SINGLE, rate, num_for(rate), offset=(i * 97) % 800, seed=i))
        return {"startup": s.startup_lines(), "runs": out}


def study_prefix(gpu):
    res = {}
    for name, flag in (("prefix_on", ["--enable-prefix-caching"]), ("prefix_off", ["--no-enable-prefix-caching"])):
        with Server(flag) as s:
            first = load(SINGLE, 4, 300, offset=0)              # first turns fill the cache
            follow = load(FOLLOW, 4, 300, offset=0)             # second turns share the first turn's prompt
            res[name] = {"first_turn": first, "follow_up": follow, "metrics": s.metrics("vllm:prefix_cache"),
                         "startup": s.startup_lines()}
    return res


def study_budget(gpu):
    res = {}
    rate = RATES[gpu][-2]
    for budget in (512, 8192):
        with Server(["--max-num-batched-tokens", str(budget)]) as s:
            load(SINGLE, 2, 20, offset=900)
            res[f"budget_{budget}"] = {"rate": rate, "result": load(SINGLE, rate, num_for(rate), offset=100)}
    return res


def study_spec(gpu):
    res = {}
    cfg = json.dumps({"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_min": 2, "prompt_lookup_max": 5})
    for name, flag in (("no_spec", []), ("ngram", ["--speculative-config", cfg])):
        with Server(flag) as s:
            load(SINGLE, 1, 10, offset=900)
            runs = {}
            for rate in (0.25, RATES[gpu][-2]):
                runs[str(rate)] = load(SINGLE, rate, 30 if rate < 1 else num_for(rate), offset=200)
            res[name] = {"runs": runs, "metrics": s.metrics("vllm:spec_decode"), "startup": s.startup_lines()}
    return res


STUDIES = {"single": study_single, "sweep": study_sweep, "prefix": study_prefix, "budget": study_budget,
           "spec": study_spec}


def run_studies(gpu: str, studies: list) -> dict:
    env = {"gpu": subprocess.getoutput("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
           "vllm": subprocess.getoutput(f"{sys.executable} -c 'import vllm; print(vllm.__version__)'"),
           "model": MODEL, "slo": SLO, "max_tokens": MAX_TOKENS, "base_args": BASE_ARGS}
    out = {"env": env}
    for name in studies:
        t0 = time.time()
        try:
            out[name] = STUDIES[name](gpu)
        except Exception as e:                                   # keep the other studies' results
            out[name] = {"error": f"{type(e).__name__}: {e}"[-4000:]}
        out[name + "_wall_s"] = round(time.time() - t0, 1)
    return out


FUNCS = {g: app.function(name=f"serve_{g.lower().replace('-', '_')}", image=serve_image, gpu=m, **COMMON)(run_studies)
         for g, m in GPUS.items()}


@app.local_entrypoint()
def main(gpus: str = "", studies: str = ""):
    sel = [g.strip() for g in gpus.split(",") if g.strip()] or list(GPUS)
    st = [s.strip() for s in studies.split(",") if s.strip()] or list(STUDIES)
    print(build_prompts.remote())
    handles = {g: FUNCS[g].spawn(g, st) for g in sel}
    outdir = HERE / "results"
    outdir.mkdir(exist_ok=True)
    for g, h in handles.items():
        res = h.get()
        (outdir / f"{g}.json").write_text(json.dumps(res, indent=2))
        print(f"wrote results/{g}.json")
