"""Chapter 11 case study, portable export row (Code 11.4): torch.export -> ONNX -> ONNX Runtime CUDA EP on an L4.

  modal run onnx/export_and_run.py
Measures prefill latency of a 512-token prompt in PyTorch FP16 eager and in ONNX Runtime FP16, lists the nodes that
ONNX Runtime placed on the CPU, and saves teacher-forced logits for the parity check. The export covers one forward
pass without a KV cache, so only prefill is timed (TPOT is not measured on this path).
"""
import json, os, sys, time
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("hpml-ch11-onnx")
hf = modal.Volume.from_name("hf-cache", create_if_missing=True)
img = (modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
       .pip_install("torch==2.14.1", "transformers>=5.16,<5.20", "onnx", "onnxscript", "onnxruntime-gpu==1.31.*",
                    "datasets", "numpy")
       .add_local_file(os.path.join(HERE, "..", "common.py"), "/root/common.py"))


@app.function(image=img, gpu="L4", secrets=[modal.Secret.from_name("huggingface")],
              volumes={"/root/.cache/huggingface": hf}, timeout=3600)
def run(ref_ids_json: str = ""):
    import numpy as np, torch, onnxruntime as ort
    sys.path.insert(0, "/root")
    from common import MODEL, MAX_PROMPT, PARITY_POSITIONS, load_prompts, chat, environment
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    model.config.use_cache = False
    prompts = [chat(tok, p) for p in load_prompts()]
    ids = tok(prompts[0], return_tensors="pt", padding="max_length", max_length=MAX_PROMPT, truncation=True).input_ids

    def t_torch():
        x = ids.cuda()
        with torch.no_grad():
            for _ in range(3): model(x)
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(10): model(x)
            torch.cuda.synchronize()
        return (time.perf_counter() - t) / 10
    torch_ms = t_torch() * 1e3

    batch, seq = torch.export.Dim("batch", max=16), torch.export.Dim("seq", max=4096)
    t = time.perf_counter()
    prog = torch.onnx.export(model, (ids.cuda(),), dynamo=True, dynamic_shapes=({0: batch, 1: seq},))
    prog.optimize(); prog.save("/tmp/gemma2b.onnx", external_data=True)
    export_s = time.perf_counter() - t
    so = ort.SessionOptions(); so.log_severity_level = 1
    so.enable_profiling = False
    sess = ort.InferenceSession("/tmp/gemma2b.onnx", so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for _ in range(3): sess.run(None, {name: ids.numpy()})
    t = time.perf_counter()
    for _ in range(10): sess.run(None, {name: ids.numpy()})
    ort_ms = (time.perf_counter() - t) / 10 * 1e3
    # nodes on CPU: rerun with verbose logging into a file is the documented way; here we count via profiling
    so2 = ort.SessionOptions(); so2.enable_profiling = True
    s2 = ort.InferenceSession("/tmp/gemma2b.onnx", so2, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    s2.run(None, {name: ids.numpy()}); prof = s2.end_profiling()
    ev = json.load(open(prof))
    cpu_nodes = sorted({e["args"].get("op_name") for e in ev if e.get("cat") == "Node"
                        and e.get("args", {}).get("provider") == "CPUExecutionProvider"})
    rows = {"l4_torch_fp16": {"ttft_ms": round(torch_ms, 2), "fallback_ops": 0, "env": environment()},
            "l4_ort_fp16": {"compile_s": round(export_s, 1), "ttft_ms": round(ort_ms, 2),
                            "fallback_ops": len(cpu_nodes), "cpu_ops": cpu_nodes, "env": environment()}}
    if ref_ids_json:
        ref = json.loads(ref_ids_json); lt, lo = [], []
        for pid, toks in zip(ref["prompt_ids"], ref["tokens"]):
            full = np.array([pid + toks[:PARITY_POSITIONS]], dtype=np.int64)
            sl = slice(len(pid) - 1, len(pid) - 1 + PARITY_POSITIONS)
            with torch.no_grad():
                lt.append(model(torch.from_numpy(full).cuda()).logits[0, sl].float().cpu().numpy())
            lo.append(sess.run(None, {name: full})[0][0, sl].astype(np.float32))
        rows["l4_torch_fp16"]["_logits"] = np.stack(lt).tobytes(); rows["l4_ort_fp16"]["_logits"] = np.stack(lo).tobytes()
    return rows


@app.local_entrypoint()
def main(ref: str = "../parity/ref/ref_ids.json"):
    import numpy as np
    sys.path.insert(0, os.path.join(HERE, ".."))
    from common import merge_json
    ref_json = ""
    if os.path.exists(ref):
        r = json.load(open(ref)); r["tokens"] = np.load(os.path.join(os.path.dirname(ref), "ref_tokens.npy")).tolist()
        ref_json = json.dumps(r)
    rows = run.remote(ref_json)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    for k, v in rows.items():
        blob = v.pop("_logits", None)
        if blob is not None:
            np.save(os.path.join(HERE, "out", f"{k}.npy"), np.frombuffer(blob, dtype=np.float32).reshape(32, 32, -1))
        merge_json(os.path.join(HERE, "..", "..", "..", "results", "case_study.json"), k, v)
    print(json.dumps(rows, indent=1, default=str)[:1500])
