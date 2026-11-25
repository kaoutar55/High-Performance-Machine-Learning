"""Chapter 6 (PyTorch 2.14.1 / Triton 3.8.0): every measured quantity in the chapter, in one script.

The kernels and the timer are loaded from chapter06_companion_2p14.ipynb (only the function definitions are
executed), so the timed code is the notebook's code. Each subcommand runs in its own process; a driver runs it
five times and `aggregate` reports the median and the min-max across processes.

    python ch6_measure_2p14.py steady   --out r1.json        # vector add, softmax, matmul, GROUP_SIZE_M
    python ch6_measure_2p14.py fused    --out f1.json        # RMSNorm (Gemma), LayerNorm, QuickGELU + dropout
    python ch6_measure_2p14.py sweep    --out s1.json        # 48-shape softmax sweep, with torch.compile
    python ch6_measure_2p14.py autotune --mode cold    --out a1.json   # first call, empty Triton cache
    python ch6_measure_2p14.py autotune --mode prewarm --out p1.json   # same, hand kernel run first (as the notebook)
    python ch6_measure_2p14.py aggregate --out final.json r*.json a*.json p*.json

Timers: "nb" is the notebook's cuda_time with its defaults (CUDA events, median of 50 calls after 10 warm-up
calls); "nb200" is the same timer with 200 calls after 25 warm-up calls (the 2026-11-23 matmul_sizes runs);
"do_bench" is triton.testing.do_bench(fn, warmup=25, rep=200), median.
GPU clocks, power limit, temperature and clock-event reasons are recorded before and after each process.
"""
import argparse
import datetime
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matmul_sizes_2p14 as ms  # noqa: E402  (loader helpers and env block)

ATOL, RTOL = ms.ATOL, ms.RTOL
WANTED = {"cuda_time", "add_kernel", "add", "softmax_kernel", "softmax", "naive_softmax",
          "matmul_kernel", "matmul", "matmul_kernel_tuned", "matmul_tuned",
          "layernorm_fwd_kernel", "layernorm", "rmsnorm_fwd_kernel", "rmsnorm",
          "fused_quick_gelu_dropout_kernel", "quick_gelu_dropout"}
# Vector-add sizes. 1,048,576 and 89,478,485 elements are the 12 MiB and 1 GiB footprints of x, y and out together
# (3 x 4 bytes per element); the H100 L2 is 50 MB, so the first fits in L2 and the second does not.
VADD_SIZES = (1_000_003, 1_048_576, 89_478_485, 100_000_000)
# Gemma-2B: hidden 2048, MLP intermediate 16384; rows 1 (one decode token) and 512 (the 512-token prompt), plus one
# large shape (128 sequences of 512 tokens). BGE-M3 (the embedder): hidden 1024, 32 passages of 128 tokens.
NORM_SHAPES = [(1, 2048), (512, 2048), (65536, 2048)]
LN_EMBEDDER_SHAPE = (4096, 1024)
ACT_SHAPES = [(1, 16384), (512, 16384), (8192, 16384)]
# The 48 shapes of results_a100/softmax_shape_sweep.json (16 shapes x FP32, FP16, BF16).
SWEEP_SHAPES = [(4096, 512), (4096, 1024), (4096, 2048), (4096, 4096), (4096, 8192), (2048, 16384),
                (8192, 1024), (8192, 2048), (16384, 512), (16384, 1024), (1024, 2048), (512, 4096),
                (4096, 1000), (4096, 3000), (4096, 100), (65536, 256)]
SMI_FIELDS = ["clocks.sm", "clocks.mem", "clocks.max.sm", "clocks.max.mem", "clocks.applications.graphics",
              "power.limit", "power.default_limit", "enforced.power.limit", "power.draw", "temperature.gpu",
              "clocks_event_reasons.active", "clocks_throttle_reasons.active", "persistence_mode"]
H100_SXM_PEAKS = {"source": "NVIDIA H100 Tensor Core GPU datasheet, H100 SXM column (dense, without sparsity)",
                  "bw_gbps": 3350.0, "fp32_tflops": 67.0, "fp16_tc_tflops": 989.4}


def gpu_state():
    """One nvidia-smi query per field, so a field the driver does not support does not lose the others."""
    out = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    for f in SMI_FIELDS:
        r = subprocess.run(["nvidia-smi", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
                            f"--query-gpu={f}", "--format=csv,noheader"], capture_output=True, text=True)
        out[f] = r.stdout.strip() if r.returncode == 0 else "n/a"
    return out


def load_defs():
    """Like matmul_sizes_2p14.load_notebook_defs, for the wider set of functions this script times."""
    import numpy as np
    import torch
    import triton
    import triton.language as tl
    import ast
    pinned, got = ("2.14.1", "3.8.0"), (torch.__version__.split("+")[0], triton.__version__)
    if got != pinned and os.environ.get("CH6_ALLOW_OTHER_STACK") != "1":
        raise SystemExit(f"expected torch/triton {pinned}, found {got} (set CH6_ALLOW_OTHER_STACK=1 to override)")
    nb = json.load(open(ms.NB))
    ns, tag = {}, os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in WANTED]
        if not defs:
            continue
        code = "\n\n".join(src[ms._start(src, n):ms._end(src, n)] for n in defs)
        path = f"/tmp/ch6_2p14_defs_{tag}_cell{i}.py"     # triton.jit reads the source from a real file
        with open(path, "w") as f:
            f.write("import torch, triton\nimport triton.language as tl\nimport numpy as np\n"
                    "dev = torch.device('cuda')\n\n" + code + "\n")
        spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        for n in defs:
            ns[n.name] = getattr(m, n.name)
    missing = WANTED - set(ns)
    if missing:
        raise SystemExit(f"notebook functions not found: {missing}")
    return ns


def do_bench_median(fn):
    import triton.testing
    try:
        return float(triton.testing.do_bench(fn, warmup=25, rep=200, return_mode="median"))
    except TypeError:   # signature change: fall back to the quantiles form
        return float(triton.testing.do_bench(fn, warmup=25, rep=200, quantiles=[0.5])[0])


def time_all(ns, fn):
    ct = ns["cuda_time"]
    m50 = ct(fn)                          # notebook defaults: iters=50, warmup=10
    m200 = ct(fn, iters=200, warmup=25)
    return {"nb": {"median_ms": m50[0], "p10_ms": m50[1], "p90_ms": m50[2]},
            "nb200": {"median_ms": m200[0], "p10_ms": m200[1], "p90_ms": m200[2]},
            "do_bench": {"median_ms": do_bench_median(fn)}}


def steady(out):
    import torch
    import torch.nn.functional as F
    state_before = gpu_state()
    ns = load_defs()
    dev = torch.device("cuda")
    res = {"kind": "steady", "env": ms.env_info(" ".join(["python"] + sys.argv)), "gpu_state_before": state_before}
    torch.manual_seed(0)

    # vector add (notebook cell 5 size, and 100M elements), FP32
    res["vadd"] = {}
    for n in VADD_SIZES:
        x = torch.rand(n, device=dev)
        y = torch.rand(n, device=dev)
        o = ns["add"](x, y)
        ok = bool(torch.allclose(o, x + y, atol=1e-6, rtol=0))
        t = time_all(ns, lambda: ns["add"](x, y))
        nbytes = 3 * n * 4
        for k in t:
            t[k]["gbps"] = nbytes / (t[k]["median_ms"] * 1e-3) / 1e9
            t[k]["tflops"] = n / (t[k]["median_ms"] * 1e-3) / 1e12
            t[k]["pct_of_3350_gbps"] = 100 * t[k]["gbps"] / H100_SXM_PEAKS["bw_gbps"]
        res["vadd"][f"n{n}"] = {"n": n, "bytes": nbytes, "footprint_mib": nbytes / 2 ** 20, "allclose": ok, **t}
        del x, y, o
        print(json.dumps({"vadd": n, "nb_ms": round(t["nb"]["median_ms"], 4), "gbps": round(t["nb"]["gbps"])}),
              flush=True)

    # softmax: naive, fused Triton, F.softmax at the notebook's two FP16 shapes (cell 9)
    res["softmax"] = {}
    for (M, N) in [(4096, 4096), (8192, 2048)]:
        x = torch.randn(M, N, device=dev, dtype=torch.float16)
        ref = torch.softmax(x.float(), dim=1)
        row = {"M": M, "N": N, "dtype": "float16"}
        for impl, fn in [("naive", lambda: ns["naive_softmax"](x)), ("fused_triton", lambda: ns["softmax"](x)),
                         ("F.softmax", lambda: F.softmax(x, dim=1))]:
            err = float((fn().float() - ref).abs().max())
            row[impl] = {"max_abs_err_vs_fp32": err, **time_all(ns, fn)}
        for tk in ("nb", "nb200", "do_bench"):
            f_ = row["fused_triton"][tk]["median_ms"]
            row.setdefault("speedup_vs_naive", {})[tk] = row["naive"][tk]["median_ms"] / f_
            row.setdefault("speedup_vs_torch", {})[tk] = row["F.softmax"][tk]["median_ms"] / f_
        row["fused_bytes"] = 2 * M * N * 2
        row["fused_gbps_nb"] = row["fused_bytes"] / (row["fused_triton"]["nb"]["median_ms"] * 1e-3) / 1e9
        res["softmax"][f"{M}x{N}"] = row
        print(json.dumps({"softmax": f"{M}x{N}", "vs_naive": round(row["speedup_vs_naive"]["nb"], 2),
                          "vs_torch": round(row["speedup_vs_torch"]["nb"], 2)}), flush=True)
        del x, ref

    # matmul: hand kernel, autotuned kernel, cuBLAS at 1,024, 2,048, 4,096 cubed; GROUP_SIZE_M 8 vs 1
    res["matmul"], res["grouping"] = {}, {}
    for n in (1024, 2048, 4096):
        a = torch.randn(n, n, device=dev, dtype=torch.float16)
        b = torch.randn(n, n, device=dev, dtype=torch.float16)
        ref = torch.matmul(a, b)
        flops = 2 * n ** 3
        row = {"size": n}
        for impl, fn in [("cublas", lambda: torch.matmul(a, b)), ("triton_hand", lambda: ns["matmul"](a, b)),
                         ("triton_autotuned", lambda: ns["matmul_tuned"](a, b))]:
            c = fn()
            torch.cuda.synchronize()
            row[impl] = {"allclose": bool(torch.allclose(c, ref, atol=ATOL, rtol=RTOL)),
                         "max_abs_err": float((c.float() - ref.float()).abs().max()), **time_all(ns, fn)}
            for tk in ("nb", "nb200", "do_bench"):
                row[impl][tk]["tflops"] = flops / (row[impl][tk]["median_ms"] * 1e-3) / 1e12
        for impl in ("triton_hand", "triton_autotuned"):
            row[impl]["pct_of_cublas"] = {tk: 100 * row["cublas"][tk]["median_ms"] / row[impl][tk]["median_ms"]
                                          for tk in ("nb", "nb200", "do_bench")}
        row["triton_autotuned"]["best_config"] = str(getattr(ns["matmul_kernel_tuned"], "best_config", None))
        res["matmul"][str(n)] = row
        if n in (1024, 4096):
            g = {}
            for gm in (8, 1):
                t = time_all(ns, lambda: ns["matmul"](a, b, GROUP_M=gm))
                g[f"GROUP_SIZE_M={gm}"] = t
            g["time_ratio_1_over_8"] = {tk: g["GROUP_SIZE_M=1"][tk]["median_ms"] / g["GROUP_SIZE_M=8"][tk]["median_ms"]
                                        for tk in ("nb", "nb200", "do_bench")}
            res["grouping"][str(n)] = g
        print(json.dumps({"matmul": n, **{k: round(row[k]["nb"]["median_ms"], 4)
                                          for k in ("cublas", "triton_hand", "triton_autotuned")}}), flush=True)
        del a, b, ref
    res["gpu_state_after"] = gpu_state()
    json.dump(res, open(out, "w"), indent=2)


def _compiled(fn):
    """torch.compile for one shape. Dynamo is reset first: the same function compiled for many shapes would
    otherwise reach the recompile limit (8 by default) and run eager without an error."""
    import torch
    torch._dynamo.reset()
    return torch.compile(fn, dynamic=False)


def _bench_impls(ns, impls, ref, atol, rtol):
    """Time each implementation with all three timers and record its error against `ref` (FP32)."""
    import torch
    row = {}
    for name, fn in impls.items():
        y = fn()                                   # first call: compiles (Triton JIT or Inductor) before timing
        torch.cuda.synchronize()
        err = float((y.float() - ref).abs().max()) if ref is not None else None
        ok = bool(torch.allclose(y.float(), ref, atol=atol, rtol=rtol)) if ref is not None else None
        row[name] = {"max_abs_err": err, "allclose": ok, **time_all(ns, fn)}
    return row


def _ratios(row, base, others):
    return {f"{o}_over_{base}": {tk: row[o][tk]["median_ms"] / row[base][tk]["median_ms"]
                                 for tk in ("nb", "nb200", "do_bench")} for o in others}


def fused(out):
    """RMSNorm (Gemma form, scale 1 + w), LayerNorm, QuickGELU + dropout: the chapter's Triton kernels against
    PyTorch eager, the library op where one exists, and torch.compile of the eager code. FP16 storage for the
    norms. QuickGELU + dropout runs in FP32: the notebook's kernel (cell 14) applies tl.exp to the loaded dtype,
    and Triton's tl.exp takes FP32 or FP64 only (FP16 input fails to compile on Triton 3.8.0)."""
    import torch
    import torch.nn.functional as F
    state_before = gpu_state()
    ns = load_defs()
    dev, dt = torch.device("cuda"), torch.float16
    res = {"kind": "fused", "env": ms.env_info(" ".join(["python"] + sys.argv)), "gpu_state_before": state_before,
           "rmsnorm_gemma": {}, "layernorm": {}, "quickgelu_dropout": {}}
    torch.manual_seed(0)

    def gemma_rmsnorm_eager(x, w, eps=1e-6):     # transformers GemmaRMSNorm.forward: FP32 math, scale (1 + w)
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())).type_as(x)

    for (R, H) in NORM_SHAPES:
        x = torch.randn(R, H, device=dev, dtype=dt)
        w = 0.1 * torch.randn(H, device=dev, dtype=dt)     # Gemma stores w with the 1 folded out
        w1 = (1.0 + w.float())                             # precomputed once per layer, outside the timed call
        ref = gemma_rmsnorm_eager(x.float(), w.float())
        cmp = _compiled(gemma_rmsnorm_eager)
        impls = {"triton": lambda: ns["rmsnorm"](x, w1, 1e-6), "eager": lambda: gemma_rmsnorm_eager(x, w),
                 "F.rms_norm": lambda: F.rms_norm(x, (H,), w1.to(dt), 1e-6), "compile": lambda: cmp(x, w)}
        row = _bench_impls(ns, impls, ref, 1e-2, 1e-2)
        row.update(_ratios(row, "triton", ["eager", "F.rms_norm", "compile"]))
        res["rmsnorm_gemma"][f"{R}x{H}"] = row
        print(json.dumps({"rmsnorm": f"{R}x{H}", **{k: round(row[k]["nb"]["median_ms"], 4) for k in impls}}),
              flush=True)

    def ln_eager(x, w, b):
        return F.layer_norm(x, (x.shape[-1],), w, b, 1e-5)
    for (R, H) in NORM_SHAPES + [LN_EMBEDDER_SHAPE]:
        x = torch.randn(R, H, device=dev, dtype=dt)
        w = torch.randn(H, device=dev, dtype=dt)
        b = torch.randn(H, device=dev, dtype=dt)
        ref = F.layer_norm(x.float(), (H,), w.float(), b.float(), 1e-5)
        cmp = _compiled(ln_eager)
        impls = {"triton": lambda: ns["layernorm"](x, w, b, 1e-5), "F.layer_norm": lambda: ln_eager(x, w, b),
                 "compile": lambda: cmp(x, w, b)}
        row = _bench_impls(ns, impls, ref, 2e-2, 1e-2)
        row.update(_ratios(row, "triton", ["F.layer_norm", "compile"]))
        row["shape_from"] = "BGE-M3 embedder" if (R, H) == LN_EMBEDDER_SHAPE else "Gemma-2B hidden size"
        res["layernorm"][f"{R}x{H}"] = row
        print(json.dumps({"layernorm": f"{R}x{H}", **{k: round(row[k]["nb"]["median_ms"], 4) for k in impls}}),
              flush=True)

    def qg_dropout_eager(x, p):
        return F.dropout(x * torch.sigmoid(1.702 * x), p, training=True)

    def qg_eager(x):
        return x * torch.sigmoid(1.702 * x)
    for (R, H) in ACT_SHAPES:
        x = torch.randn(R, H, device=dev, dtype=torch.float32)
        ref = qg_eager(x.float())
        cd, cq = _compiled(qg_dropout_eager), _compiled(qg_eager)
        entry = {}
        # p = 0.1, training: outputs differ by RNG, so correctness is checked on the Triton output itself
        impls = {"triton": lambda: ns["quick_gelu_dropout"](x, 0.1, 123), "eager": lambda: qg_dropout_eager(x, 0.1),
                 "compile": lambda: cd(x, 0.1)}
        row = _bench_impls(ns, impls, None, 0, 0)
        y = ns["quick_gelu_dropout"](x, 0.1, 123).float()
        kept = y != 0
        row["triton"]["kept_fraction"] = float(kept.float().mean())
        row["triton"]["survivors_allclose"] = bool(torch.allclose(y[kept], (ref / 0.9)[kept], atol=1e-5, rtol=1e-3))
        row.update(_ratios(row, "triton", ["eager", "compile"]))
        entry["p0.1_training"] = row
        # p = 0, the inference case: the activation alone (the Triton kernel still draws the random numbers)
        impls = {"triton": lambda: ns["quick_gelu_dropout"](x, 0.0, 0), "eager": lambda: qg_eager(x),
                 "compile": lambda: cq(x)}
        row = _bench_impls(ns, impls, ref, 1e-2, 1e-2)
        row.update(_ratios(row, "triton", ["eager", "compile"]))
        entry["p0_inference"] = row
        res["quickgelu_dropout"][f"{R}x{H}"] = entry
        print(json.dumps({"quickgelu_dropout": f"{R}x{H}",
                          "p0.1": {k: round(entry["p0.1_training"][k]["nb"]["median_ms"], 4) for k in impls}}),
              flush=True)
    res["gpu_state_after"] = gpu_state()
    json.dump(res, open(out, "w"), indent=2)


def sweep(out):
    """The 48-shape softmax sweep of results_a100/softmax_shape_sweep.json, plus torch.compile(F.softmax)."""
    import torch
    import torch.nn.functional as F
    state_before = gpu_state()
    ns = load_defs()
    res = {"kind": "sweep", "env": ms.env_info(" ".join(["python"] + sys.argv)), "gpu_state_before": state_before,
           "rows": {}}

    def sm(x):
        return F.softmax(x, dim=1)
    for dname, dt in (("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)):
        for (M, N) in SWEEP_SHAPES:
            x = torch.randn(M, N, device="cuda", dtype=dt)
            ref = torch.softmax(x.float(), dim=1)
            atol = 1e-5 if dt == torch.float32 else 3e-2      # the A100 sweep's tolerances
            cmp = _compiled(sm)
            impls = {"naive": lambda: ns["naive_softmax"](x), "fused_triton": lambda: ns["softmax"](x),
                     "F.softmax": lambda: sm(x), "compile": lambda: cmp(x)}
            row = _bench_impls(ns, impls, ref, atol, 0)
            row.update(_ratios(row, "fused_triton", ["naive", "F.softmax", "compile"]))
            res["rows"][f"{dname}_{M}x{N}"] = row
            del x, ref
            torch.cuda.empty_cache()
        print(json.dumps({"sweep_done": dname}), flush=True)
    wins = {}
    for tk in ("nb", "do_bench"):
        for base in ("F.softmax", "compile"):
            wins[f"fused_faster_than_{base}_{tk}"] = sum(r[f"{base}_over_fused_triton"][tk] > 1 for r in res["rows"].values())
    res["wins_of_48"] = wins
    res["gpu_state_after"] = gpu_state()
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(wins), flush=True)


def autotune(out, mode):
    cache = os.environ.get("TRITON_CACHE_DIR")
    if not cache:
        raise SystemExit("set TRITON_CACHE_DIR to a new, empty directory for a first-call measurement")
    if os.path.isdir(cache) and os.listdir(cache):
        raise SystemExit(f"TRITON_CACHE_DIR {cache} is not empty")
    state_before = gpu_state()
    import torch
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()                      # CUDA context outside the timed region
    ns = load_defs()
    if mode == "prewarm":                         # as the notebook: cell 22 runs the hand kernel before cell 26
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        ns["matmul"](x, x)
        torch.cuda.synchronize()
    n = 2048
    torch.manual_seed(0)
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    c = ns["matmul_tuned"](a, b)
    torch.cuda.synchronize()
    first = (time.perf_counter() - t0) * 1e3
    med, p10, p90 = ns["cuda_time"](lambda: ns["matmul_tuned"](a, b))   # the notebook's steady state (cell 26)
    res = {"kind": f"autotune_{mode}", "env": ms.env_info(" ".join(["python"] + sys.argv)),
           "gpu_state_before": state_before, "size": n, "mode": mode, "first_call_ms": first,
           "steady_median_ms": med, "steady_p10_ms": p10, "steady_p90_ms": p90, "ratio": first / med,
           "allclose": bool(torch.allclose(c, torch.matmul(a, b), atol=ATOL, rtol=RTOL)),
           "best_config": str(getattr(ns["matmul_kernel_tuned"], "best_config", None)),
           "gpu_state_after": gpu_state()}
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("mode", "first_call_ms", "steady_median_ms", "best_config")}), flush=True)


# ---------------------------------------------------------------------------------------------- aggregation
def _flat(d, prefix=""):
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from _flat(v, p)
        else:
            yield p, v


def _summ(vals):
    nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(nums) == len(vals) and nums:
        return {"median": statistics.median(nums), "min": min(nums), "max": max(nums), "n": len(nums)}
    if all(isinstance(v, bool) for v in vals):
        return {"all": all(vals), "n": len(vals)}
    return {"values": sorted(set(map(str, vals))), "n": len(vals)}


def aggregate(out, files):
    runs = [json.load(open(f)) for f in files]
    groups = {}
    for r in runs:
        groups.setdefault(r["kind"], []).append(r)
    final = {"date": datetime.date.today().isoformat(), "env": runs[0]["env"],
             "method": __doc__.strip().split("\n\n", 2)[-1],
             "n_processes": {k: len(v) for k, v in groups.items()}, "summary": {}, "gpu_states": [],
             "raw_files": [os.path.basename(f) for f in files]}
    final["env"].pop("command", None)
    final["env"]["drivers_seen"] = sorted({r["env"]["driver"] for r in runs})
    for kind, rs in groups.items():
        keys = {}
        for r in rs:
            body = {k: v for k, v in r.items() if k not in ("env", "gpu_state_before", "gpu_state_after", "kind")}
            for p, v in _flat(body):
                keys.setdefault(p, []).append(v)
        final["summary"][kind] = {p: _summ(v) for p, v in keys.items()}
        for r in rs:
            final["gpu_states"].append({"kind": kind, "before": r["gpu_state_before"], "after": r["gpu_state_after"]})
    # roofline points on the H100 SXM datasheet roofs, from the median steady-state timings (notebook timer)
    s = final["summary"].get("steady", {})
    if s:
        def med(k): return s[k]["median"]
        def mm(n): return 2 * n ** 3 / (3 * n * n * 2)
        pts = {"vector_add": (1 / 12, 1_000_003 / (med("vadd.n1000003.nb.median_ms") * 1e-3) / 1e12),
               "fused_softmax": (1.25, 5 * 4096 * 4096 / (med("softmax.4096x4096.fused_triton.nb.median_ms") * 1e-3) / 1e12),
               "matmul_hand_1024": (mm(1024), med("matmul.1024.triton_hand.nb.tflops")),
               "cublas_1024": (mm(1024), med("matmul.1024.cublas.nb.tflops")),
               "matmul_autotuned_2048": (mm(2048), med("matmul.2048.triton_autotuned.nb.tflops"))}
        pk = dict(H100_SXM_PEAKS) if "H100" in final["env"]["gpu_name"] else {"source": "set the peaks for this GPU"}
        if "bw_gbps" in pk:
            pk["ridge_fp32"] = pk["fp32_tflops"] * 1e3 / pk["bw_gbps"]
            pk["ridge_fp16"] = pk["fp16_tc_tflops"] * 1e3 / pk["bw_gbps"]
        final["roofline"] = {"peaks": pk, "points": {k: {"ai": a, "tflops": t} for k, (a, t) in pts.items()}}
    json.dump(final, open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["steady", "fused", "sweep", "autotune", "aggregate"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["cold", "prewarm"], default="cold")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    if a.what == "steady":
        steady(a.out)
    elif a.what == "fused":
        fused(a.out)
    elif a.what == "sweep":
        sweep(a.out)
    elif a.what == "autotune":
        autotune(a.out, a.mode)
    else:
        aggregate(a.out, a.files)
