"""Run Chapter 7's measured studies on a two-GPU Modal container and write modal_results.txt.

    modal run chapter07/modal_run.py              # analytics on CPU, then the DDP and FSDP studies on 2 GPUs
    modal run chapter07/modal_run.py::gpu_only    # only the DDP and FSDP studies

GPU type: set CH07_GPU (default "A10G:2", the configuration the notebook's measured cells came from).
"""
import os
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).parent
GPU = os.environ.get("CH07_GPU", "A10G:2")
SCRIPTS = ["ddp_train.py", "fsdp_train.py", "mem_wall.py", "comm_cost.py", "bubble.py", "tp_mlp.py", "moe_routing.py"]

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.14.1", "numpy")
for name in SCRIPTS:
    image = image.add_local_file(HERE / name, f"/root/ch07/{name}")
app = modal.App("hpml-ch07", image=image)

DDP_RUNS = [
    ("ddp_small_1gpu", 1, "--steps 60 --warmup 15 --batch-size 32 --d-model 1024"),
    ("ddp_small_2gpu", 2, "--steps 60 --warmup 15 --batch-size 32 --d-model 1024"),
    ("ddp_big_1gpu", 1, "--steps 60 --warmup 15 --batch-size 64 --d-model 4096"),
    ("ddp_big_2gpu", 2, "--steps 60 --warmup 15 --batch-size 64 --d-model 4096"),
    ("ddp_2gpu_nosampler", 2, "--steps 60 --warmup 15 --batch-size 32 --d-model 1024 --no-sampler"),
]
FSDP_RUNS = [
    ("fsdp_2gpu", 2, "--steps 50 --warmup 15"),
    ("fsdp_2gpu_ckpt", 2, "--steps 50 --warmup 15 --checkpoint"),
]


def _torchrun(script: str, nproc: int, args: str) -> str:
    cmd = f"torchrun --standalone --nproc_per_node={nproc} {script} {args}"
    r = subprocess.run(cmd, shell=True, cwd="/root/ch07", capture_output=True, text=True)
    lines = [l for l in (r.stdout + r.stderr).splitlines()
             if l.startswith(("[rank", "RESULT", "Traceback")) or "Error" in l]
    return "\n".join(lines) + "\n"


def _section(title: str) -> str:
    return "=" * 70 + f"\n{title}\n" + "=" * 70 + "\n"


@app.function(gpu=GPU, timeout=3600)
def run_gpu_studies() -> str:
    out = [_section("DDP")]
    for name, nproc, args in DDP_RUNS:
        out.append(f"\n----- {name} -----\n" + _torchrun("ddp_train.py", nproc, args) + "\n\n")
    out.append(_section("FSDP"))
    for name, nproc, args in FSDP_RUNS:
        out.append(f"\n----- {name} -----\n" + _torchrun("fsdp_train.py", nproc, args) + "\n\n")
    return "".join(out)


@app.function(timeout=600)
def run_analytics() -> str:
    out = [_section("ANALYTICS")]
    for name in ("mem_wall.py", "comm_cost.py", "bubble.py", "tp_mlp.py", "moe_routing.py"):
        r = subprocess.run(["python", name], cwd="/root/ch07", capture_output=True, text=True)
        out.append(f"\n----- {name} -----\n{r.stdout}{r.stderr}\n")
    return "".join(out)


@app.local_entrypoint()
def main():
    text = run_analytics.remote() + "\n" + run_gpu_studies.remote()
    Path("modal_results.txt").write_text(text)
    print(text)


@app.local_entrypoint()
def gpu_only():
    text = run_gpu_studies.remote()
    Path("modal_results.txt").write_text(text)
    print(text)
