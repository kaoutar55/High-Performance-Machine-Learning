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
