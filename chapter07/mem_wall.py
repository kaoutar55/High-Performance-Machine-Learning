"""Model-state memory: bytes per parameter for common training regimes, and when a model stops fitting.

Model state = parameters + gradients + (optional) FP32 master weights + optimizer states. Activations are not
counted here; they depend on batch size and sequence length (see Chapter 7, "Why a Single GPU Runs Out").
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Regime:
    name: str
    param_bytes: int     # bytes per parameter for the working copy of the weights
    grad_bytes: int      # bytes per parameter for the gradient buffer
    master_bytes: int    # bytes per parameter for an FP32 master copy (0 if none)
    optim_bytes: int     # bytes per parameter for optimizer state (Adam: two moments)

    @property
    def bytes_per_param(self) -> int:
        return self.param_bytes + self.grad_bytes + self.master_bytes + self.optim_bytes


REGIMES = [
    # 2 (FP16/BF16 weights) + 2 (grads) + 4 (FP32 master) + 8 (FP32 Adam m and v) = 16, the ZeRO paper's count
    Regime("Mixed-precision Adam baseline", 2, 2, 4, 8),
    # BF16 weights and grads, FP32 master copy, Adam moments kept in BF16: 2 + 2 + 4 + 4 = 12
    Regime("BF16 params, low-precision optimizer state", 2, 2, 4, 4),
    # Everything in FP32, no separate master copy: 4 + 4 + 0 + 8 = 16
    Regime("Full FP32 Adam", 4, 4, 0, 8),
    # BF16 weights and grads, no master copy, 8-bit Adam moments: 2 + 2 + 0 + 2 = 6
    Regime("BF16 params, 8-bit optimizer state", 2, 2, 0, 2),
]


def model_state_gb(params_billion: float, bytes_per_param: float, shards: int = 1) -> float:
    """Model-state memory per GPU in GB (1e9 bytes) with linear sharding over `shards` GPUs (ZeRO-3 / FSDP)."""
    return params_billion * bytes_per_param / shards


def min_gpus(params_billion: float, bytes_per_param: float, gpu_gb: float) -> int:
    """Smallest number of GPUs whose combined memory holds the model state under linear sharding."""
    return max(1, math.ceil(model_state_gb(params_billion, bytes_per_param) / gpu_gb - 1e-9))


def build_table(sizes_billion=(1, 7, 70), gpu_gb: float = 80, shard_count: int = 8):
    """One row per (regime, model size): replicated and sharded model state, and whether it fits."""
    rows = []
    for r in REGIMES:
        for size in sizes_billion:
            replicated = model_state_gb(size, r.bytes_per_param)
            rows.append({
                "regime": r.name,
                "bytes_per_param": r.bytes_per_param,
                "params_billion": size,
                "replicated_gb": replicated,
                "fits_one_80gb_replicated": replicated <= gpu_gb,
                "per_gpu_gb_8way": model_state_gb(size, r.bytes_per_param, shard_count),
                "min_80gb_gpus_linear_shard": min_gpus(size, r.bytes_per_param, gpu_gb),
            })
    return rows


if __name__ == "__main__":
    for row in build_table():
        print(row)
