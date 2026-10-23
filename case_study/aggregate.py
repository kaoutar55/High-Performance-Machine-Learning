"""Combine repeated runs of the case study into one result per GPU.

Overhead-bound measurements (batch-1 decode, batch-1 encoding) depend on the host machine Modal
assigns, so they can differ by 2x between runs; compute- and bandwidth-bound measurements move a
few percent. Each repeat runs in its own container. This module merges the repeats for one GPU:

  * median(runs) -> a result with the same structure as a single <gpu>.json, every number replaced
    by the median across repeats (so make_tables.py, principles.py and the chapter build scripts
    work unchanged);
  * spread(runs) -> the same structure with [min, max] for every number.

Strings and booleans are taken from the first repeat. Lists are merged element by element when the
repeats have the same length.
"""
import statistics


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _merge(xs, fn):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    if all(isinstance(x, dict) for x in xs):
        keys = list(dict.fromkeys(k for x in xs for k in x))
        return {k: _merge([x.get(k) for x in xs], fn) for k in keys}
    if all(isinstance(x, list) for x in xs) and len({len(x) for x in xs}) == 1:
        return [_merge(list(items), fn) for items in zip(*xs)]
    if all(_num(x) for x in xs):
        return fn([float(x) for x in xs])
    return xs[0]


def median(runs):
    return _merge(runs, statistics.median)


def spread(runs):
    return _merge(runs, lambda v: [min(v), max(v)])


def relative_spread(lo_hi, mid):
    """(max - min) / median, or None."""
    if not lo_hi or not mid:
        return None
    return (lo_hi[1] - lo_hi[0]) / mid


LAB_KEY = "chapter02_lab"


def lab_device(res, notebook_text=None):
    """Device the Chapter 2 lab container ran on (the lab and the experiments run in separate
    containers, so they can land on different hardware)."""
    dev = ((res.get(LAB_KEY) or {}).get("summary") or {}).get("gpu")
    if dev:
        return dev
    if notebook_text:
        for name in ("H200", "H100 80GB HBM3", "H100 PCIe", "A100-SXM4-80GB", "A100 80GB PCIe", "A100-SXM4-40GB", "Tesla T4"):
            if name in notebook_text and not (name == "Tesla T4" and notebook_text.count(name) <= 4
                                              and any(n in notebook_text for n in ("H100", "H200", "A100"))):
                return name
    return None


def merge_repeats(runs, want, lab_devices=None):
    """Median and spread across repeats, keeping each container's part only if it ran on a device
    containing `want`. Returns (merged, spread, per_repeat_views, log)."""
    lab_devices = lab_devices or [None] * len(runs)
    log, exp_ok, lab_ok, views = [], [], [], []
    for r, ld in zip(runs, lab_devices):
        ed = (r.get("env") or {}).get("gpu", "")
        e_good = (not want) or want in ed
        l_good = (not want) or (ld is None and e_good) or (ld is not None and want in ld)
        v = {k: x for k, x in r.items() if (k == LAB_KEY and l_good) or (k != LAB_KEY and e_good)}
        if not e_good:
            log.append(f"repeat {r.get('repeat')}: experiments ran on {ed}; excluded")
        if not l_good:
            log.append(f"repeat {r.get('repeat')}: Chapter 2 lab ran on {ld}; excluded")
        if e_good:
            exp_ok.append({k: x for k, x in r.items() if k != LAB_KEY})
        if l_good and LAB_KEY in r:
            lab_ok.append(r[LAB_KEY])
        views.append(v)
    merged = median(exp_ok) if len(exp_ok) > 1 else (dict(exp_ok[0]) if exp_ok else {})
    sp = spread(exp_ok) if len(exp_ok) > 1 else None
    if lab_ok:
        merged[LAB_KEY] = median(lab_ok) if len(lab_ok) > 1 else lab_ok[0]
        if sp is not None and len(lab_ok) > 1:
            sp[LAB_KEY] = spread(lab_ok)
    merged.pop("repeat", None); merged.pop("source", None)
    merged["repeats"] = len(exp_ok)
    merged["lab_repeats"] = len(lab_ok)
    if exp_ok:
        merged["env"]["gpu"] = exp_ok[0]["env"]["gpu"]
    return merged, sp, views, log
