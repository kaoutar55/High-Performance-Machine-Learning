# Chapter 8 Knowledge check: solutions

High-Performance Inference and Serving

Section names refer to the chapter. In this chapter 1 KB is 1,024 bytes.

## Question 1

Why is the decode phase of a batch-one request memory-bound, and how does increasing the batch size change its
arithmetic intensity? Why does this argument not apply to the attention part of the step?

**Answer.** A batch-one decode step reads the whole model to perform about two floating-point operations per
parameter, roughly 1 FLOP per byte in 16-bit precision, far below the case-study GPUs' ridge points (153 to
295 FLOP/byte). With batch B, one weight read serves B rows, so the intensity of the weight matrix multiplications
grows about in proportion to B. Attention reads each request's own KV cache, so batching adds bytes and FLOPs in the
same proportion, and attention decode stays at about 1 FLOP per byte (times the GQA group size) at any batch size
for MHA, GQA and MQA models.

**Explanation.** *Prefill and decode phases*, Figure 8.1, and *KV cache traffic in the decode step*; MLA is the
exception, because many query heads share one cached latent.

## Question 2

Define TTFT, ITL, TPOT and goodput. Why can a service with a low mean TPOT still appear slow to users?

**Answer.** TTFT is the time from sending the request to the first output token (queueing plus prefill). ITL is the
time between two consecutive streamed tokens. TPOT is (end-to-end latency − TTFT) / (output tokens − 1), the mean
ITL of one request. Goodput is the number of requests per second that meet their latency objectives, both TTFT and
TPOT. A low mean TPOT hides stalls: a request with a 30 ms TPOT can have a 400 ms gap when another request's long
prefill runs in the same iteration, and a long queue raises TTFT.

**Explanation.** *Latency metrics for streaming requests*, Table 8.1; Principle #24 asks for percentiles next to
the mean.

## Question 3

Compute the KV cache per token of Llama-3.1-8B (32 layers, 8 KV heads, head dimension 128) in BF16 and FP8. How many
8K-token requests fit in 60 GB of cache memory in each case?

**Answer.** 2 × 32 × 8 × 128 × 2 bytes = 131,072 bytes (128 KB) per token in BF16, and 64 KB in FP8. An 8,192-token
request takes 1 GiB in BF16 and 0.5 GiB in FP8. 60 GB (60 × 10⁹ bytes, 55.9 GiB) holds about 55 such requests in
BF16 and about 111 in FP8 (60 and 120 if the 60 GB are read as GiB).

**Explanation.** KV bytes per token = 2 × layers × KV heads × head dimension × bytes per element (*The KV cache*);
Chapter 9's Table 9.6 gives the same 128 KiB and 64 KiB and about 55 and 110 requests for the 56 GiB that vLLM
leaves on an 80 GB GPU.

## Question 4

What problem does selective batching solve in an iteration-level scheduler, and which operator still needs
per-request handling?

**Answer.** Under continuous batching the requests of one iteration are at different stages (prefill or decode) with
different cached lengths, so their tensors do not stack into one rectangular batch. Selective batching flattens the
tokens of all requests for the token-wise operators (linear layers, MLP, normalizations), which then run as one
large matrix multiplication. Attention still needs per-request structure, because each token attends to its own
request's earlier tokens.

**Explanation.** Later attention libraries, such as FlashAttention's variable-length kernels and FlashInfer, handle
the ragged sequences in one kernel launch (*Continuous batching*, Figure 8.5).

## Question 5

Chunked prefill lowers p99 inter-token latency but can raise TTFT. Explain the trade-off and the setting that
controls it.

**Answer.** Splitting a long prompt into chunks that fit the iteration's token budget lets the running decodes
continue every iteration, which bounds their inter-token latency; the long prompt's prefill then spans several
iterations, so its TTFT grows. The token budget controls the balance (`--max-num-batched-tokens` in vLLM): a larger
budget finishes prefills sooner, a smaller one keeps decode steps shorter.

**Explanation.** *Chunked prefill* and *Continuous batching*, Figure 8.6, Principle #25.

## Question 6

Describe the three kinds of KV memory waste in contiguous allocation and how paging reduces each.

**Answer.** Internal fragmentation: memory reserved for the maximum length that the request does not write. Reserved
slots: slots the request will write later, unused until it reaches them. External fragmentation: gaps between slabs
of different sizes that are too small for a new request. Paging allocates fixed-size blocks (16 tokens by default in
vLLM) on demand from anywhere in the pool, so the only token-slot waste is the unfilled part of each request's last
block, at most 15 slots, and a new request fits in any free blocks.

**Explanation.** The vLLM authors found that only 20.4% to 38.2% of the KV cache memory of earlier systems held token
states (*Memory waste in contiguous allocation*, *PagedAttention and block tables*, Figure 8.10).

## Question 7

A system prompt contains the current date and time on its first line. What happens to prefix caching, and how would
you fix it?

**Answer.** Prefix caching reuses only an exact prefix, and a changing token early in the prompt invalidates every
block after it, so every request misses the cache and pays the full prefill. Move the changing content to the end of
the prompt, after the stable system prompt, tool schemas and retrieved documents, or drop the time from the prompt.

**Explanation.** *Prefix caching*; Principle #27 asks you to order prompts so that the stable part comes first.

## Question 8

Explain the online softmax used by FlashAttention. Why does the result equal standard attention exactly?

**Answer.** The kernel loops over tiles of K and V and keeps, per query, the running maximum m, the running sum ℓ of
exponentials relative to m, and the unnormalized output. When a tile raises the maximum, the earlier sum and output
are multiplied by exp(m_old − m_new) before the new tile is added; after the last tile the output is divided by ℓ.
The rescaling makes every earlier term relative to the final maximum, so the algebra is the same as the full-row
softmax and the result differs only by floating-point rounding.

**Explanation.** Code 8.2 checks the tiled computation against `torch.softmax(q @ k.T / 8, dim=-1) @ v` in FP64 with
a maximum difference of about 1e-16 (*Tiling and the online softmax*).

## Question 9

Why does FlashAttention parallelize poorly for single-request decode at long context, and how does Flash-Decoding
fix it?

**Answer.** FlashAttention parallelizes over batch, heads and query blocks; with one query token there is one query
block per head, so a 32-head model at batch one launches about 32 thread blocks on a GPU with 108 or 132 SMs, and
each block reads the whole cache alone. Flash-Decoding splits the cache along the keys into chunks processed by
separate thread blocks, each producing a partial output and the log-sum-exp of its scores, and a second kernel
rescales and combines the partials.

**Explanation.** The authors reported up to 8× faster decoding at very long sequences for CodeLlama-34B (*Attention
kernels for decode*, Figure 8.15).

## Question 10

With an acceptance rate of 0.7 and γ = 3, how many tokens does each target pass produce on average? Why can
speculative decoding lower throughput at large batch sizes?

**Answer.** E = (1 − α^(γ+1)) / (1 − α) = (1 − 0.7⁴) / 0.3 = 0.7599 / 0.3 ≈ 2.53 tokens per target pass. At large
batch, each step already processes many tokens; multiplying them by γ + 1 moves the step past the ridge point, where
verification adds compute time, and rejected draft tokens are work that other requests could have used.

**Explanation.** In Figure 8.17 the modeled gain for Llama-3.1-8B on an H100 with 512-token contexts falls below one
at about 200 concurrent sequences, and the vLLM team reported 1.4× to 1.8× slowdowns at high request rates
(*The draft-and-verify algorithm*, *Speculative decoding at large batch sizes*, Principle #29).

## Question 11

Why does an MoE model with 37B active parameters need much larger decode batches than a 37B dense model to use its
GPUs efficiently?

**Answer.** In an MoE layer with E experts and top-k routing, each expert sees on average B × k / E of the batch's
tokens, so each expert's matrix multiplication has E/k times lower arithmetic intensity than a dense layer at the
same batch (one thirty-second for DeepSeek-V3, 8 routed of 256). Bringing DeepSeek-V3's expert GEMMs near the H100's
FP8 ridge needs a few hundred tokens per expert per step, a global decode batch of thousands of tokens.

**Explanation.** Memory is set by the total parameters and compute by the active ones (*Arithmetic intensity of MoE
decode*, Principle #30).
