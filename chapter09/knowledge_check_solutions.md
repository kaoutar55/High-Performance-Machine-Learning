# Chapter 9 Knowledge check: solutions

Model Compression: Quantization and Sparsity

Section names refer to the chapter. GB is 10⁹ bytes and GiB is 2³⁰ bytes, as in the chapter.

## Question 1

Llama-3.1-8B has about 16 GB of BF16 weights. Why does quantizing the linear layers to INT4 with groups of 128
reduce the weight bytes by about 2.8 times when the bit width falls by 4 times? Why is the bound lower still for
Gemma-2B?

**Answer.** Two terms stay large. The 4-bit format stores one FP16 scale per 128 weights, 4.125 bits per weight, and
the embedding table and the output projection stay in 16 bits. Code 9.1 counts 16.06 GB in BF16 and 5.70 GB in INT4
for Llama-3.1-8B, a 2.82× reduction. Gemma-2B (2.42×, 5.01 to 2.07 GB) gains less because its 16-bit output
projection, about 1 GB for a 256,000-token vocabulary, is 21% of the bytes of each step, against 13% for the output
projection and embedding table of Llama-3.1-8B.

**Explanation.** *Bytes and FLOPs in LLM inference*, Code 9.1. The ratios bound the speedup of a memory-bound decode
step only if the quantized kernel reaches the same effective bandwidth and the fixed per-step costs do not change.

## Question 2

Quantize [-1.8, 0.0, 0.7, 2.5] to asymmetric INT8 over the range [-1.8, 2.5]. Give the scale, the zero point and the
dequantized values, and state the largest rounding error.

**Answer.** S = 4.3 / 255 ≈ 0.016863 and Z = round(-128 + 1.8 / S) = round(-128 + 106.74) = -21. With
q = clamp(round(r / S) + Z, -128, 127): q = -128, -21, 21, 127. Dequantized with r̂ = S × (q − Z): -1.8043, 0.0,
0.7082, 2.4957. The errors are 0.0043, 0, 0.0082 and 0.0043; the largest is 0.0082, for 0.7, within the bound of
half a step (S / 2 ≈ 0.0084).

**Explanation.** The formulas and the scale and zero point for this range are those of *Scale and zero point*; the
real value 0 maps exactly to -21, which keeps zero padding and masked positions exact.

## Question 3

An FP8 checkpoint of Llama-3.1-8B loads on an A100 and an H100. On which GPU does it speed up prefill, and why? What
would you use on the other GPU instead?

**Answer.** On the H100, which has FP8 Tensor Cores: FP8 W8A8 runs at twice the BF16 rate (1,979 against 989 dense
TFLOP/s), which speeds up compute-bound prefill. The A100 has no FP8 arithmetic and converts the FP8 weights to 16
bits inside the kernel, so the checkpoint saves memory and runs at the 16-bit rate. On the A100, use INT8 W8A8 to
speed up the arithmetic, or W4A16 for decode latency.

**Explanation.** *Formats for quantized inference* and *Format selection by hardware*, Principle #34.

## Question 4

You serve Llama-3.1-8B on an A100 with 4K-token prompts, and its capacity is limited by the KV cache. Rank FP8 W8A8,
INT8 W8A8, W4A16 and an FP8 KV cache by the capacity gain you expect, and justify the ranking.

**Answer.** 1. FP8 KV cache: it halves the bytes of every cached token, so the same cache memory holds twice the
tokens (Table 9.10), which is the largest gain in concurrency when the cache is the limit. 2. INT8 W8A8: the 8-bit
weights free memory for the cache (8.52 GiB of weights against 15.00 GiB, a cache larger by 6.5 GiB on about
56 GiB), and INT8 is the only 8-bit format that speeds up the A100's arithmetic, which raised Llama-3.1-8B's capacity
on the A100 from 12 to 20 requests per second with short prompts. 3. W4A16: it frees the most weight memory (5.36 GiB
of weights, a cache larger by 9.6 GiB) but speeds up only memory-bound decode at small batch; on the H100
long-context workload it added 17% to the cache and left capacity at 3 requests per second, the BF16 value.
4. FP8 W8A8: the A100 has no FP8 arithmetic, so the checkpoint runs weight-only, with the memory saving of an 8-bit
format, no faster arithmetic and the cost of converting the weights inside the kernel; it left capacity unchanged in
the case study.

**Explanation.** Compress the tensor that limits the workload (Principle #31): with 4K-token prompts the KV cache
limits concurrency, weight formats add cache only through the memory they free, and only formats with native
arithmetic raise the rate at which a GPU completes requests within the objectives (*KV cache bytes and capacity*,
*Format selection by hardware*, *Results: memory and latency*, *Results: capacity and long context*). On the H100
long-context workload, the FP8 KV cache raised capacity from 3 to 4 requests per second and FP8 W8A8 with an FP8 KV
cache to 6, the highest rate tested.
