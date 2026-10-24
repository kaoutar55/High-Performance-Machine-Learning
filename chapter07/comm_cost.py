"""Communication cost of gradient synchronization: ring all-reduce against a parameter server.

N is the gradient payload in bytes and P the number of ranks. The alpha-beta model charges alpha seconds per
message and 1/bw seconds per byte on each link.
"""


def ring_allreduce_bytes_per_rank(n_bytes: float, p: int) -> float:
    """Bytes each rank sends in a ring all-reduce: reduce-scatter plus all-gather, 2N(P-1)/P."""
    return 2.0 * n_bytes * (p - 1) / p


def param_server_bytes(n_bytes: float, p: int) -> float:
    """Bytes through a single parameter server: P-1 workers each send N and receive N, 2N(P-1)."""
    return 2.0 * n_bytes * (p - 1)


def ring_allreduce_time(n_bytes: float, p: int, alpha_s: float, bw_bytes_s: float) -> float:
    """Alpha-beta time of a ring all-reduce: 2(P-1) latency steps plus 2(P-1)/P * N / bw."""
    return 2 * (p - 1) * alpha_s + ring_allreduce_bytes_per_rank(n_bytes, p) / bw_bytes_s


def allgather_time(n_bytes: float, p: int, alpha_s: float, bw_bytes_s: float) -> float:
    """Alpha-beta time of an all-gather or a reduce-scatter: half of an all-reduce."""
    return (p - 1) * alpha_s + (p - 1) / p * n_bytes / bw_bytes_s


def bus_bandwidth(n_bytes: float, p: int, seconds: float) -> float:
    """nccl-tests bus bandwidth of an all-reduce: algorithm bandwidth times 2(P-1)/P."""
    return n_bytes / seconds * 2 * (p - 1) / p


def crossover_ratio(p: int) -> float:
    """Parameter-server volume divided by per-rank ring volume. It equals P for any payload."""
    return param_server_bytes(1.0, p) / ring_allreduce_bytes_per_rank(1.0, p)


if __name__ == "__main__":
    gib = 1024 ** 3
    for p in (2, 8, 64):
        print(p, ring_allreduce_bytes_per_rank(gib, p) / 1e9, param_server_bytes(gib, p) / 1e9)
