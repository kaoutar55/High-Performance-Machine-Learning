"""Swappable experiment settings for the book's case study.

Everything in this file is expected to change over the book's life: GPUs, models, sizes and package
versions. The durable claims the book teaches live in principles.py and must keep holding whatever
is configured here. To refresh the book for a new GPU or model, edit this file and re-run:

    modal run case_study/modal_case_study.py                 # every GPU in GPUS
    modal run case_study/modal_case_study.py --gpus T4       # a subset, comma-separated
"""

# GPUs to run on (Modal GPU names). The first is the book's reference platform.
GPUS = ["A100-80GB", "H100", "T4"]

# Datasheet peaks used for Roofline placement, matched against the device name the GPU reports.
# Dense (no sparsity) 16-bit Tensor Core TFLOP/s and memory bandwidth in GB/s.
GPU_SPECS = [
    # (substring of torch.cuda.get_device_name(), label, fp16_tensor_tflops, mem_bw_gbs, memory)
    ("T4", "NVIDIA T4", 65.0, 320.0, "16 GB GDDR6"),
    ("A100-SXM4-80GB", "NVIDIA A100 80GB SXM", 312.0, 2039.0, "80 GB HBM2e"),
    ("A100 80GB PCIe", "NVIDIA A100 80GB PCIe", 312.0, 1935.0, "80 GB HBM2e"),
    ("A100-SXM4-40GB", "NVIDIA A100 40GB SXM", 312.0, 1555.0, "40 GB HBM2"),
    ("H100 80GB HBM3", "NVIDIA H100 SXM", 989.0, 3350.0, "80 GB HBM3"),
    ("H100 PCIe", "NVIDIA H100 PCIe", 756.0, 2000.0, "80 GB HBM2e"),
]

# The RAG case study's components, defined by role. Swap a model by changing its id.
MODELS = {
    "generator": "google/gemma-2b-it",   # small decoder LLM (~2.5B parameters with embeddings)
    "embedder": "BAAI/bge-m3",           # production-grade multilingual embedder (568M parameters)
}
DTYPE = "float16"                        # supported on every GPU listed; BF16 needs compute capability 8.0+

# Workload sizes.
PROMPT_TOKENS = 512        # retrieved context + question
NEW_TOKENS = 20            # answer length used throughout the book
DECODE_BATCHES = [1, 4, 16, 64]
EMBED_CHUNK_TOKENS = 512
EMBED_BATCHES = [1, 8, 32, 64]
EMBED_CHUNKS = 256         # chunks timed per batch size (64 for batch 1)
TIMING = {"warmup": 2, "iters": 10}   # 10 (was 5): batch-1 decode on fast GPUs is noisy

# Python packages for the GPU image. For an edition release, pin exact versions here: after a
# successful run, copy the versions recorded under "env" in the results files.
# Pinned to the stack of the first three-GPU run (23 Oct 2026); the rest are recorded per run under
# env["packages"] and should be pinned from there before the edition is released.
PIP_PACKAGES = [
    "torch==2.14.1", "transformers==5.20.2", "accelerate", "sentence-transformers==6.2.0", "faiss-cpu", "datasets",
    "numpy", "pandas", "matplotlib", "nbclient", "nbformat", "ipykernel",
]
