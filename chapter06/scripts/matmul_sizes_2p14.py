"""Chapter 6 (PyTorch 2.14.1 / Triton 3.8.0 port): Triton matmul against cuBLAS at 1,024, 2,048 and 4,096 cubed, and the autotuning first call.

The kernels and the timing helper are loaded from chapter06_companion_2p14.ipynb itself: only the function
definitions of the cells that define cuda_time, matmul_kernel/matmul and matmul_kernel_tuned/matmul_tuned
are executed, so the measured code is the notebook's code.

    python matmul_sizes_2p14.py sizes --out sizes.json          # part a and c
    python matmul_sizes_2p14.py autotune --out autotune.json    # part b: one first call in this process
FP16 inputs, FP32 accumulation (tl.dot accumulator), FP16 output. Timing: the notebook's cuda_time
(CUDA events, median) with 200 timed calls after 25 warm-up calls, cross-checked with triton.testing.do_bench.
"""
import argparse
import ast
import datetime
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.environ.get("CH6_NOTEBOOK", os.path.join(HERE, "..", "chapter06_companion_2p14.ipynb"))
ATOL, RTOL = 1e-1, 1e-2          # the notebook's tolerances (cells 22 and 26)


def load_notebook_defs():
    import numpy as np
    import torch
    import triton
    import triton.language as tl
    pinned, got = ("2.14.1", "3.8.0"), (torch.__version__.split("+")[0], triton.__version__)
    if got != pinned and os.environ.get("CH6_ALLOW_OTHER_STACK") != "1":
        raise SystemExit(f"expected torch/triton {pinned}, found {got} (set CH6_ALLOW_OTHER_STACK=1 to override)")
    nb = json.load(open(NB))
    ns = {"torch": torch, "np": np, "triton": triton, "tl": tl, "dev": torch.device("cuda")}
    wanted = {"cuda_time", "matmul_kernel", "matmul", "matmul_kernel_tuned", "matmul_tuned"}
    found = set()
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
        if defs:
            # triton.jit reads the function's source text, so the definitions are written to a real file
            code = "\n\n".join(src[_start(src, n):_end(src, n)] for n in defs)
            path = f"/tmp/ch6_nb_defs_{'_'.join(n.name for n in defs)}.py"
            open(path, "w").write("import torch, triton\nimport triton.language as tl\nimport numpy as np\n"
                                  "dev = torch.device('cuda')\n\n" + code + "\n")
            import importlib.util
            spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            for n in defs:
                ns[n.name] = getattr(m, n.name); found.add(n.name)
    missing = wanted - found
    if missing:
        raise SystemExit(f"notebook functions not found: {missing}")
    return ns


def _start(src, node):
    lines = src.splitlines(keepends=True)
    first = min([d.lineno for d in node.decorator_list] + [node.lineno])
    return sum(len(x) for x in lines[:first - 1])


def _end(src, node):
    lines = src.splitlines(keepends=True)
    return sum(len(x) for x in lines[:node.end_lineno])


def cpu_model():
    """CPU model from /proc/cpuinfo. Modal's sandbox reports "model name: unknown"; then record vendor,
    family, model number, clock and the AVX-512 flags, which identify the CPU generation."""
    info = {}
    try:
        for ln in open("/proc/cpuinfo"):
            if not ln.strip():
                break
            k, _, v = ln.partition(":")
            info[k.strip()] = v.strip()
    except Exception:  # noqa: BLE001
        return "unknown"
    name = info.get("model name", "unknown")
    if name and name != "unknown":
        return name
    flags = info.get("flags", "").split()
    avx = ",".join(f for f in ("avx512f", "avx512_vnni", "avx512_bf16", "amx_tile") if f in flags) or "no avx512"
    return (f"model name hidden by the sandbox; {info.get('vendor_id', '?')} family {info.get('cpu family', '?')} "
            f"model {info.get('model', '?')}, {info.get('cpu MHz', '?')} MHz, {avx}")


def env_info(cmd):
    import torch
    import triton
    q = subprocess.getoutput("nvidia-smi --query-gpu=name,memory.total,driver_version,power.max_limit "
                             "--format=csv,noheader").split(",")
    name = q[0].strip()
    cpu = cpu_model()
    return {"date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "command": cmd, "gpu_name": name, "gpu_memory": q[1].strip(), "driver": q[2].strip(),
            "gpu_power_max_limit": q[3].strip(),
            "gpu_form_factor": "SXM" if "SXM" in name else ("PCIe" if "PCIe" in name else "unknown"),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "driver_cuda_version": subprocess.getoutput("nvidia-smi | grep -oE 'CUDA (UMD )?Version: [0-9.]+'")
            .split(':')[-1].strip(),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda, "triton": triton.__version__,
            "python": platform.python_version(), "host_cpu": cpu, "cpu_count_visible": os.cpu_count(),
            "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR", "default ~/.triton/cache")}


def sizes(out, iters=200, warmup=25):
    import torch
    import triton
    ns = load_notebook_defs()
    cuda_time, matmul, matmul_tuned = ns["cuda_time"], ns["matmul"], ns["matmul_tuned"]
    res = {"env": env_info(" ".join(["python"] + sys.argv)), "rows": [], "grouping_4096": None,
           "timing": f"notebook cuda_time: CUDA events, median of {iters} calls after {warmup} warm-up calls; "
                     "do_bench: triton.testing.do_bench median (warmup 25 ms, rep 200 ms)"}
    torch.manual_seed(0)
    for n in (1024, 2048, 4096):
        a = torch.randn(n, n, device="cuda", dtype=torch.float16)
        b = torch.randn(n, n, device="cuda", dtype=torch.float16)
        ref = torch.matmul(a, b)
        flops = 2 * n ** 3
        impls = {"cublas": lambda: torch.matmul(a, b), "triton_hand": lambda: matmul(a, b),
                 "triton_autotuned": lambda: matmul_tuned(a, b)}
        row = {"size": n}
        for k, fn in impls.items():
            c = fn(); torch.cuda.synchronize()
            ok = bool(torch.allclose(c, ref, atol=ATOL, rtol=RTOL))
            err = float((c.float() - ref.float()).abs().max())
            med, p10, p90 = cuda_time(fn, iters=iters, warmup=warmup)
            db = float(triton.testing.do_bench(fn, warmup=25, rep=200, return_mode="median"))
            row[k] = {"median_ms": med, "p10_ms": p10, "p90_ms": p90, "do_bench_median_ms": db,
                      "tflops": flops / (med * 1e-3) / 1e12, "allclose": ok, "max_abs_err": err}
        for k in ("triton_hand", "triton_autotuned"):
            row[k]["pct_of_cublas"] = 100 * row["cublas"]["median_ms"] / row[k]["median_ms"]
            row[k]["pct_of_cublas_do_bench"] = 100 * row["cublas"]["do_bench_median_ms"] / row[k]["do_bench_median_ms"]
        cfg = getattr(ns["matmul_kernel_tuned"], "best_config", None)
        row["triton_autotuned"]["best_config"] = str(cfg)
        res["rows"].append(row)
        print(json.dumps({"size": n, **{k: round(row[k]["median_ms"], 4) for k in impls}}), flush=True)
        if n == 4096:
            g = {}
            for gm in (8, 1):
                med, p10, p90 = cuda_time(lambda: matmul(a, b, GROUP_M=gm), iters=iters, warmup=warmup)
                g[f"GROUP_SIZE_M={gm}"] = {"median_ms": med, "p10_ms": p10, "p90_ms": p90,
                                           "tflops": flops / (med * 1e-3) / 1e12}
            res["grouping_4096"] = g
            print(json.dumps({k: round(v["median_ms"], 4) for k, v in g.items()}), flush=True)
        json.dump(res, open(out, "w"), indent=2)


def autotune(out, prewarm):
    import torch
    torch.zeros(1, device="cuda"); torch.cuda.synchronize()     # CUDA context outside the timed region
    ns = load_notebook_defs()
    cuda_time, matmul_tuned = ns["cuda_time"], ns["matmul_tuned"]
    if prewarm:   # as in the notebook: cell 22 runs the hand kernel at 1,024 before cell 26
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        ns["matmul"](x, x); torch.cuda.synchronize()
    n = 2048
    torch.manual_seed(0)
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    c = matmul_tuned(a, b); torch.cuda.synchronize()
    first = time.perf_counter() - t0
    med, p10, p90 = cuda_time(lambda: matmul_tuned(a, b), iters=200, warmup=25)
    res = {"env": env_info(" ".join(["python"] + sys.argv)), "size": n, "prewarm_hand_kernel": prewarm, "first_call_ms": first * 1e3,
           "steady_median_ms": med, "steady_p10_ms": p10, "steady_p90_ms": p90,
           "allclose": bool(torch.allclose(c, torch.matmul(a, b), atol=ATOL, rtol=RTOL)),
           "best_config": str(getattr(ns["matmul_kernel_tuned"], "best_config", None))}
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("first_call_ms", "steady_median_ms")}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["sizes", "autotune"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--prewarm", action="store_true")
    ap.add_argument("--iters", type=int, default=200, help="timed calls per measurement (notebook default: 50)")
    ap.add_argument("--warmup", type=int, default=25, help="warm-up calls (notebook default: 10)")
    a = ap.parse_args()
    sizes(a.out, a.iters, a.warmup) if a.what == "sizes" else autotune(a.out, a.prewarm)
