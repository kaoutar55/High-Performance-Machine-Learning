"""KV cache bytes per token and token capacity from a Hugging Face model configuration (Code 8.x in the chapter).

    python kv_calc.py google/gemma-2b-it meta-llama/Llama-3.1-8B-Instruct --gpu-gb 80 --weight-gb 5.0
"""
import argparse

from transformers import AutoConfig


def kv_bytes_per_token(model_id, dtype_bytes=2):
    c = AutoConfig.from_pretrained(model_id)
    c = getattr(c, "text_config", c)                     # multimodal configs nest the language model
    heads = c.num_attention_heads
    kv_heads = getattr(c, "num_key_value_heads", None) or heads
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // heads
    return 2 * c.num_hidden_layers * kv_heads * head_dim * dtype_bytes


def kv_token_capacity(model_id, gpu_gb, weight_gb, util=0.90, overhead_gb=2.0, dtype_bytes=2):
    free = gpu_gb * util - weight_gb - overhead_gb       # activations, CUDA graphs, workspace
    return int(free * 1e9 // kv_bytes_per_token(model_id, dtype_bytes))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--gpu-gb", type=float, default=80)
    ap.add_argument("--weight-gb", type=float, default=None, help="weights in GB (default: skip capacity)")
    ap.add_argument("--dtype-bytes", type=int, default=2, help="2 for BF16/FP16 KV, 1 for FP8")
    a = ap.parse_args()
    for m in a.models:
        b = kv_bytes_per_token(m, a.dtype_bytes)
        line = f"{m}: {b:,} bytes/token, {b * 4096 / 2**30:.2f} GiB per 4K-token sequence"
        if a.weight_gb is not None:
            line += f", capacity {kv_token_capacity(m, a.gpu_gb, a.weight_gb, dtype_bytes=a.dtype_bytes):,} tokens"
        print(line)
