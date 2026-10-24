"""Pipeline-parallel bubble fraction (total-time convention).

A synchronous P-stage pipeline with M microbatches per step runs for M + P - 1 slots, of which P - 1 are fill
and drain: the bubble is (P - 1) / (M + P - 1). Interleaved 1F1B with v model chunks per device shrinks the idle
slots by v: (P - 1) / (v*M + P - 1).
"""


def bubble_fraction(p: int, m: int) -> float:
    return (p - 1) / (m + p - 1)


def bubble_fraction_interleaved(p: int, m: int, v: int = 2) -> float:
    return (p - 1) / (v * m + p - 1)


def microbatches_for_target(p: int, target: float, v: int = 1) -> int:
    """Smallest M that keeps the bubble at or below `target` (for example 0.10)."""
    m = 1
    while (p - 1) / (v * m + p - 1) > target + 1e-12:
        m += 1
    return m


if __name__ == "__main__":
    for p in (4, 8, 16):
        print(p, microbatches_for_target(p, 0.10))
