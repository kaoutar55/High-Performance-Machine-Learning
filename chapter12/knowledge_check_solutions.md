# Chapter 12 Final quiz: solutions

The Mature ML System

The quiz covers the whole book. Questions 1 to 10 name the chapter they come from; questions 11 to 15 are
calculations from Chapter 12. Section names refer to the chapter named in each answer.

## Question 1

A pipeline spends 30% of its time in retrieval and 70% in generation. What is the largest end-to-end speedup that a
tenfold faster retriever can give, and which law states the limit? (Chapters 1 and 2)

**Answer.** 1 / (0.7 + 0.3 / 10) = 1 / 0.73 ≈ 1.37×, by Amdahl's Law. Even an infinitely fast retriever gives at most
1 / 0.7 ≈ 1.43×.

**Explanation.** The speedup of the whole is limited by the share of time that the change does not touch (Chapter 2,
*Amdahl's Law*); Chapter 3 applies the same bound to the non-decode stages (Table 3.11).

## Question 2

Why does full fine-tuning with Adam in mixed precision need about 16 bytes per parameter, and what does that imply
for an 8B model on one 80 GB GPU? (Chapters 3 and 10)

**Answer.** 2 bytes for the 16-bit weights, 2 for their gradients, 4 for an FP32 master copy and 8 for the two FP32
Adam moments. An 8B model needs about 128 GB of model state before activations, more than one 80 GB GPU holds.

**Explanation.** Chapter 3, Table 3.4 (Llama 3 8B: 16 + 16 + 96 GB, and about 174 GB with activations and
FlashAttention); Chapter 10, *Memory of full fine-tuning*.

## Question 3

What is a graph break in torch.compile, and why can it reduce the speedup of compilation? (Chapter 4)

**Answer.** When Dynamo reaches Python it cannot trace into a tensor graph, it compiles the part it captured, runs
the untraceable code as residual bytecode in eager mode and resumes compiling afterward. Each break ends a compiled
region, so fusion stops there and per-kernel launch overhead returns; many small breaks leave a model that is
compiled and still pays most of the eager-mode cost.

**Explanation.** Chapter 4, *TorchDynamo: capturing a graph from ordinary Python* and Principle #8.

## Question 4

A decode step at batch 1 reads 16 GB of weights on a GPU with 3.35 TB/s of memory bandwidth and 989 TFLOP/s of BF16
compute. Is it compute-bound or memory-bound, and what is the lower bound on its duration? (Chapter 5)

**Answer.** Memory-bound: the step performs about 1 FLOP per byte, and the ridge point is 989 / 3.35 ≈ 295 FLOP/byte.
The lower bound is 16 GB / 3.35 TB/s ≈ 4.8 ms.

**Explanation.** Chapter 5, *Dense linear layers: decode and prefill* and *Performing Roofline analysis*.

## Question 5

What does tiling a matrix multiplication in Triton change about the traffic between global memory and on-chip
memory? (Chapter 6)

**Answer.** Each program instance loads a BLOCK_SIZE_M × BLOCK_SIZE_K tile of A and a BLOCK_SIZE_K × BLOCK_SIZE_N tile
of B from global memory once and uses them for BLOCK_SIZE_M × BLOCK_SIZE_N × BLOCK_SIZE_K multiply-accumulates on
chip. Each loaded byte is reused across more arithmetic as the tiles grow, so global-memory traffic falls relative to
the work and the arithmetic intensity rises toward the compute roof.

**Explanation.** Chapter 6, *Tiled Matrix Multiplication*.

## Question 6

In FSDP, which collective operations run in the forward and backward passes, and how does their volume compare with
DDP's all-reduce? (Chapter 7)

**Answer.** The forward pass all-gathers each layer's parameters before using them; the backward pass all-gathers
them again and reduce-scatters the gradients. Each moves about N bytes per GPU per step, about 3N in total against
2N for DDP's all-reduce, 50% more.

**Explanation.** Chapter 7, *What the sharding costs in communication*, Figure 7.12.

## Question 7

Why does the KV cache, and not the weights, limit the number of concurrent requests at long context? How does paged
attention change the number that fit? (Chapter 8)

**Answer.** The weights occupy memory once per GPU, while the KV cache occupies memory for every token of every
running request, so at long context the cache fills the memory left after the weights. Paged attention allocates the
cache in fixed-size blocks on demand, so a request wastes at most one partial block; earlier contiguous allocators
used only 20.4% to 38.2% of the cache memory for token states, and vLLM's throughput rose 2× to 4× over them at the
same latency.

**Explanation.** Chapter 8, Principles #26 and #27, *Memory waste in contiguous allocation* and *PagedAttention and
block tables*.

## Question 8

An FP8 checkpoint runs on an A100 and an H100. On which GPU does it speed up prefill, and why? (Chapter 9)

**Answer.** On the H100, whose Tensor Cores run FP8 at twice the BF16 rate (1,979 against 989 dense TFLOP/s). The
A100 has no FP8 arithmetic and converts the weights to 16 bits inside the kernel, so prefill runs at the 16-bit rate.

**Explanation.** Chapter 9, *Formats for quantized inference* and *Format selection by hardware*.

## Question 9

When should new knowledge be added to the retrieval index, and when should the model be fine-tuned? (Chapter 10)

**Answer.** New facts go into the retrieval index; fine-tuning is for new behaviors: how the model uses its context,
formats its output or decides to abstain.

**Explanation.** Chapter 10, Principle #41: updating an index takes minutes and leaves the model unchanged, and it
outperformed fine-tuning for injecting new facts in the cited studies and in the case study's knowledge probe.

## Question 10

Why does a dataflow accelerator with weights in on-chip SRAM reach higher single-user decode speed than an HBM-based
GPU, and what does it need more of to serve a 70B model? (Chapter 11)

**Answer.** Batch-1 decode is bound by the bandwidth of the memory that holds the weights, and SRAM bandwidth is far
higher: about 150 TB/s per Groq LP30 against 3.35 TB/s of HBM on an H100. SRAM capacity per chip is small (about
500 MB), so a 70B model needs many more chips: about 140 for the FP8 weights alone, before any KV cache.

**Explanation.** Chapter 11, *The Roofline model for SRAM-resident designs* and *What limits dataflow designs*.

## Question 11

Estimate the KV cache transfer time for a 2,048-token prompt of Llama-3.1-8B (128 KiB per token in BF16) over a
200 Gb/s link.

**Answer.** 2,048 × 128 KiB = 256 MiB, 268 MB. A 200 Gb/s link carries 25 GB/s, so the transfer takes about
268 MB / 25 GB/s ≈ 11 ms.

**Explanation.** The same calculation in *Disaggregated prefill and decode* gives 1.34 GB and about 27 ms for a
4,096-token Llama-3.1-70B prompt over 400 Gb/s; engines hide most of it by sending each layer's cache when that layer
finishes.

## Question 12

One GPU completes 0.4 tasks per second within the objectives, and the service runs at 50% of that rate on average.
How many GPU-seconds does each task use, and what is the cost per task at a price of P per GPU-hour?

**Answer.** GPUs / (goodput × utilization) = 1 / (0.4 × 0.5) = 5.0 GPU-seconds per task (2.5 at full utilization).
The cost per task is P × 1 / (0.4 × 0.5 × 3,600) = P / 720.

**Explanation.** *Cost accounting: from GPU-hours to cost per task*, Code 12.1: the cost per task rises in inverse
proportion to utilization (Principle #47).

## Question 13

Compute the KV bytes per token in BF16 for a model with 32 layers and head dimension 128 with 32, 8 and 1 KV heads.

**Answer.** 2 × 32 × KV heads × 128 × 2 bytes = 16,384 bytes per KV head: 524,288 bytes (512 KiB) with 32 KV heads,
131,072 bytes (128 KiB) with 8, and 16,384 bytes (16 KiB) with 1.

**Explanation.** These are the MHA, GQA and MQA designs of *Architecture choices that set inference cost*; Chapter 8
gives 512 KB for Llama-2-7B and 128 KB for Llama-3.1-8B.

## Question 14

An agent's calls are spread round-robin over four replicas, each with its own prefix cache. One replica reaches a
prefix hit rate of 71%, and 48% of the prompt tokens are a shared system prompt and tool prefix. Ignoring eviction,
what hit rate does round-robin give, and what policy raises it?

**Answer.** Only the part outside the shared prefix follows the 1/N prediction: s + (71 − s) / N =
48 + 23 / 4 ≈ 54%. Prefix-aware (cache-aware) routing, which sends each call to the replica that holds the longest
prefix of its prompt, raises it.

**Explanation.** In experiment E8 with four replicas at 8 tasks per second, round-robin, session-affinity and
prefix-aware routing reached hit rates of 53%, 71% and 83% (*Case Study: The Mature Multimodal, Agentic RAG System*,
*Cache-aware routing*, Principle #49).

## Question 15

A workload has 2,000-token prompts and 2,000-token outputs at 10 requests per second. Using the method of Code 12.3,
explain whether you expect it to need more prefill or more decode capacity, and what would make disaggregation worth
its cost.

**Answer.** More decode capacity. With Code 12.3's assumptions (Llama-3.1-8B in BF16 on H100 GPUs, TPOT objective
50 ms), prefill needs 10 × 2,000 × 16.06 GFLOP / (0.5 × 989 TFLOP/s) ≈ 0.65 GPUs. Decode at a mean context of
3,000 tokens fits a batch of 146 in the KV cache, with a Roofline step of about 22 ms, so it needs
10 × 2,000 × 0.022 s / 146 ≈ 3.0 GPUs, about 4.6 times the prefill capacity. Disaggregation pays when strict
objectives for both phases cannot be met on shared GPUs, when the prefill-to-decode ratio is far from what one pool
provides, or when the two phases can run on different, cheaper hardware; for a single node with a mixed workload,
chunked prefill on shared GPUs is usually the better first step.

**Explanation.** *Disaggregated prefill and decode*, Code 12.3 and Figure 12.4 (the reasoning workload, with long
outputs, needs twenty times more decode than prefill); Principle #48.
