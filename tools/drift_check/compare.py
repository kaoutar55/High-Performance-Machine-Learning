"""Compare a new case-study result for one GPU against the reference run.

    python tools/drift_check/compare.py NEW.json REFERENCE.json [--threshold 0.20] [--report out.md]

Flags
  * any principle (case_study/principles.py) whose verdict changed (holds -> FAILS / n/a, or back);
  * any experiment that now reports an error;
  * any measured number that moved by more than the threshold (default 20%) under
    decode_batching, embedder.throughput, compiled_decode.*.decode_ms_per_token and
    chapter02_lab.summary.

Numbers from the Chapter 2 lab's hand-written decode loop are host-sensitive (two runs of the same
stack differed by up to 2x on a T4), so they are reported as warnings and do not fail the check.
Exit status: 1 if anything other than a warning was found, else 0.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "case_study"))
import principles  # noqa: E402

ID_KEYS = ("batch", "dtype")                     # identify rows in lists of dicts
SKIP = {"batch", "dtype", "chunks", "prompt_tokens", "n", "first_call_s", "dynamo_graphs_compiled",
        "repeat", "repeats"}
WARN_ONLY = ("chapter02_lab.summary",)


def leaves(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in SKIP:
                yield from leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            tag = ",".join(f"{k}={v[k]}" for k in ID_KEYS if isinstance(v, dict) and k in v) or str(i)
            yield from leaves(v, f"{path}[{tag}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, float(obj)


def watched(res):
    out = {}
    for key in ("decode_batching", "chapter02_lab"):
        sub = res.get(key) or {}
        if key == "chapter02_lab":
            sub = {"summary": sub.get("summary") or {}}
        out.update(dict(leaves(sub, key)))
    out.update(dict(leaves({"throughput": (res.get("embedder") or {}).get("throughput", [])}, "embedder")))
    for name, v in (res.get("compiled_decode") or {}).items():
        if isinstance(v, dict) and "decode_ms_per_token" in v:
            out[f"compiled_decode.{name}.decode_ms_per_token"] = float(v["decode_ms_per_token"])
    return out


def errors(res):
    errs = [k for k in ("embedder", "decode_batching") if "error" in (res.get(k) or {})]
    for name, v in ((res.get("compiled_decode") or {}).get("runs") or {}).items():
        if "error" in v and name != "decode_compile_reduce_overhead":   # known crash, see CHANGES_SINCE_PUBLICATION.md
            errs.append(f"compiled_decode.{name}")
    for name, v in (res.get("extra") or {}).items():
        if isinstance(v, dict) and "error" in v and name != "nsys":
            errs.append(f"extra.{name}")
    return errs


def compare(new, ref, threshold):
    fails, warns = [], []
    gpu = new.get("requested_gpu", "GPU")
    pv_new = {r["id"]: r["results"][gpu]["holds"] for r in principles.evaluate({gpu: new})}
    pv_ref = {r["id"]: r["results"][gpu]["holds"] for r in principles.evaluate({gpu: ref})}
    for pid in pv_ref:
        if pv_new.get(pid) != pv_ref[pid]:
            fails.append(f"principle `{pid}`: {pv_ref[pid]} -> {pv_new.get(pid)}")
    for e in sorted(set(errors(new)) - set(errors(ref))):
        fails.append(f"new error in `{e}`")
    a, b = watched(new), watched(ref)
    for path in sorted(set(a) & set(b)):
        if b[path] == 0:
            continue
        rel = (a[path] - b[path]) / abs(b[path])
        if abs(rel) > threshold:
            line = f"`{path}`: {b[path]:g} -> {a[path]:g} ({100 * rel:+.0f}%)"
            (warns if path.startswith(WARN_ONLY) else fails).append(line)
    missing = sorted(set(b) - set(a))
    if missing:
        fails.append(f"{len(missing)} measurements missing from the new run, e.g. `{missing[0]}`")
    return fails, warns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new"); ap.add_argument("reference")
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    new, ref = json.loads(pathlib.Path(args.new).read_text()), json.loads(pathlib.Path(args.reference).read_text())
    fails, warns = compare(new, ref, args.threshold)
    env_n, env_r = new.get("env") or {}, ref.get("env") or {}
    lines = ["# Drift check", "",
             f"New run: {env_n.get('gpu')} · PyTorch {env_n.get('torch')} · transformers {env_n.get('transformers')}",
             f"Reference: {env_r.get('gpu')} · PyTorch {env_r.get('torch')} · transformers {env_r.get('transformers')}",
             f"Threshold: {100 * args.threshold:.0f}%", "",
             f"## Result: {'DRIFT FOUND' if fails else 'no drift'}", ""]
    if fails:
        lines += ["## Failures", ""] + [f"- {x}" for x in fails] + [""]
    if warns:
        lines += ["## Warnings (host-sensitive, do not fail the check)", ""] + [f"- {x}" for x in warns] + [""]
    report = "\n".join(lines) + "\n"
    if args.report:
        pathlib.Path(args.report).write_text(report)
    print(report)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
