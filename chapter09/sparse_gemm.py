"""2:4 semi-structured sparsity on the GPU: when does the 2x Tensor Core peak become a real speedup?

Times y = x @ W^T for the four linear layers of Llama-3.1-8B, dense FP16 against the same weights pruned to 2:4
and stored with torch.sparse.to_sparse_semi_structured (cuSPARSELt, and the CUTLASS kernels), for token counts
from 1 (one decode step) to 16,384 (a large prefill batch).

    python sparse_gemm.py                                  # on an A100 or H100
    modal run chapter09/sparse_gemm.py                     # A100 and H100 on Modal, plus the CPU listings

vLLM 0.32 no longer serves 2:4 checkpoints, so this kernel-level benchmark is how the chapter measures the
hardware side of sparsity.
"""
import json
import pathlib
import subprocess
import sys

SHAPES = {"qkv_proj": (6144, 4096), "o_proj": (4096, 4096), "gate_up_proj": (28672, 4096), "down_proj": (4096, 14336)}
TOKENS = [1, 8, 32, 128, 512, 2048, 8192, 16384]


def bench(fn, iters=50):
    import torch
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end) / iters                  # ms


def run():
    import torch
    import torch.nn.functional as F
    from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured
    torch.manual_seed(0)
    rows = []
    for name, (n, k) in SHAPES.items():
        w = torch.randn(n, k, dtype=torch.float16, device="cuda")
        g = w.reshape(-1, 4)                                # 2:4 mask: keep the 2 largest |w| of every 4
        drop = g.abs().argsort(dim=1)[:, :2]
        w = g.scatter(1, drop, 0).reshape(n, k)
        variants = {"dense": w}
        for backend, cutlass in (("cusparselt", False), ("cutlass", True)):
            try:
                SparseSemiStructuredTensor._FORCE_CUTLASS = cutlass
                variants[backend] = to_sparse_semi_structured(w)
            except Exception as e:
                rows.append({"layer": name, "backend": backend, "error": str(e)[:300]})
        for m in TOKENS:
            x = torch.randn(m, k, dtype=torch.float16, device="cuda")
            ref = F.linear(x, w)
            row = {"layer": name, "n": n, "k": k, "tokens": m}
            for vname, wt in variants.items():
                try:
                    y = F.linear(x, wt)
                    ok = torch.allclose(y.float(), ref.float(), atol=1e-1, rtol=1e-2)
                    ms = bench(lambda: F.linear(x, wt))
                    row[vname] = {"ms": ms, "tflops": 2 * m * n * k / ms / 1e9, "correct": bool(ok)}
                except Exception as e:                       # e.g. shapes a backend does not support
                    row[vname] = {"error": str(e)[:200]}
            for b in ("cusparselt", "cutlass"):
                if "ms" in row.get(b, {}) and "ms" in row["dense"]:
                    row[b]["speedup"] = row["dense"]["ms"] / row[b]["ms"]
            rows.append(row)
    return {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "rows": rows}


def table(res):
    print(res["gpu"], "torch", res["torch"])
    print("layer          tokens  dense ms  cuSPARSELt x  CUTLASS x")
    for r in res["rows"]:
        if "tokens" not in r:
            print(r); continue
        sp = lambda b: f"{r[b]['speedup']:.2f}" if "speedup" in r.get(b, {}) else "  -"
        print(f"{r['layer']:14s} {r['tokens']:6d}  {r['dense'].get('ms', float('nan')):8.4f}  {sp('cusparselt'):>12s}  "
              f"{sp('cutlass'):>9s}")


if __name__ == "__main__" and "modal" not in sys.modules:
    table(run())

# ------------------------------------------------------------------------------------------------ Modal
try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    HERE = pathlib.Path(__file__).resolve().parent
    image = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.14.1", "numpy")
             .add_local_dir(str(HERE), "/root/ch09", ignore=["results", "*.ipynb", "__pycache__"]))
    app = modal.App("hpml-ch09-sparse")

    @app.function(image=image, gpu="A100-80GB", timeout=1800)
    def gemm_a100():
        name = subprocess.getoutput("nvidia-smi --query-gpu=name --format=csv,noheader")
        if "SXM4" not in name:
            return {"error": f"wrong GPU variant: {name}", "retry": True}
        return run()

    @app.function(image=image, gpu="H100!", timeout=1800)
    def gemm_h100():
        return run()

    @app.function(image=image, cpu=8, memory=16384, timeout=1800)
    def listings():
        out = {}
        for script in ("quant_basics.py", "smoothquant_demo.py", "prune24.py"):
            r = subprocess.run([sys.executable, script], cwd="/root/ch09", capture_output=True, text=True)
            out[script] = {"stdout": r.stdout, "stderr": r.stderr[-2000:], "returncode": r.returncode}
        out["torch"] = subprocess.getoutput(f"{sys.executable} -c 'import torch; print(torch.__version__)'")
        return out

    @app.local_entrypoint()
    def main(what: str = "gemm,listings"):
        res_dir = HERE / "results"; res_dir.mkdir(exist_ok=True)
        if "listings" in what:
            out = listings.remote()
            (res_dir / "listings.json").write_text(json.dumps(out, indent=2))
            for k, v in out.items():
                print("==", k); print(v["stdout"] if isinstance(v, dict) else v)
        if "gemm" in what:
            for tag, fn in (("A100", gemm_a100), ("H100", gemm_h100)):
                for _ in range(6):
                    r = fn.remote()
                    if not r.get("retry"):
                        break
                (res_dir / f"sparse_gemm_{tag}.json").write_text(json.dumps(r, indent=2))
                if "rows" in r:
                    table(r)
                else:
                    print(tag, r)
