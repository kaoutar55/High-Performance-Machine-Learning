# Chapter 3 Knowledge check: solutions

PyTorch Internals and Execution Performance

Section names refer to the chapter.

## Question 1

In an inference service, the output tensors of your model have a grad_fn, and peak GPU memory is higher than the
weights and one batch of activations explain. What is the root cause, what is the fix, and what does the fix save
in time and memory?

**Answer.** Autograd is still recording: the forward pass runs without `torch.no_grad()` or
`torch.inference_mode()`, so it builds the graph and keeps activations for a backward pass that does not run. Wrap
inference in `torch.inference_mode()` (or `torch.no_grad()`). For one Gemma-2B-IT prefill (FP16, 512 tokens) the
memory the forward pass adds fell from 1.95 GB to 0.26 GB on every GPU, and the time fell by 1.17× on the A100 and
1.20× on the H100 with `no_grad`, with no gain on the T4.

**Explanation.** An output tensor that carries a `grad_fn` shows that autograd is still recording (Principle #6).
The time saving appears only where the per-operation CPU work is on the critical path; on the T4 the 512-token
prefill keeps the GPU busy long enough to hide it, while the memory saving appears on every GPU (*Avoiding autograd
at inference*, Table 3.5).

## Question 2

GPU utilization is 18% over 100 training steps, the profiler shows 80% of step time in `aten::copy_` with high CPU
and low CUDA percentages, and the data-fraction timer reads 60%. Using the decision tree in Figure 3.13, name the
bottleneck class and the three most effective DataLoader changes to try first, in order.

**Answer.** A data-pipeline (data-loading) bottleneck. Try, in order: raise `num_workers` (sweep 0, 2, 4 and 8 and
keep the fastest), set `pin_memory=True` together with `non_blocking=True` transfers, and raise
`prefetch_factor` (2 to 4), with `persistent_workers=True` for multi-epoch training.

**Explanation.** Low utilization sends you to the data-fraction timer, and a reading above 10% means DataLoader
tuning will improve throughput; a large wait share with low GPU utilization is the signature of a data-pipeline
bottleneck (*Profiling the data pipeline gap*, Figure 3.13). Table 3.6 names `num_workers` the most effective
parameter and `num_workers` and `pin_memory` as the two that carry the most weight; on the A100, four workers with
pinned memory raised throughput 1.6× and cut the wait share from 39% to 5% (*DataLoader configuration*, Table 3.7).

## Question 3

Explain the activation memory cost of autograd for a 32-layer transformer (d=4096, h=32, s=2048, b=4, BF16): the
residual-stream shorthand, why the true activation memory is more than an order of magnitude larger, the total
training memory with and without FlashAttention from Table 3.4, and the effect and cost of activation
recomputation.

**Answer.** The shorthand L × b × s × d × 2 bytes = 32 × 4 × 2,048 × 4,096 × 2 bytes, about 2 GB. The forward pass
also saves the query, key and value projections, the attention scores, the softmax output, the feed-forward
intermediates and the layer-norm inputs; the score term grows as b × h × s × s, quadratic in sequence length, and
is the largest piece at long context. Table 3.4 gives about 46 GB of activations with FlashAttention and about
110 GB without, for totals of about 174 GB and 238 GB; FlashAttention removes the materialized score matrices
(about 1 GiB each for the pre-softmax scores and the softmax output per layer, 64 GiB over 32 layers).
Checkpointing every layer keeps only each layer's input, about the 2 GB residual-stream figure, at the cost of
about one extra forward pass, 30% to 40% more execution time in the cited runs.

**Explanation.** Activations are the only memory consumer that grows with batch size and sequence length, and the
gap between the two bars of Figure 3.8 is the score matrix that FlashAttention does not materialize; with
checkpoints every √L layers, peak activation memory drops from O(L) to O(√L) (*Considering the cost of activation
memory*, Table 3.4).

## Question 4

Using the launch-overhead worked example, estimate the step time and the fraction of it the GPU is busy for a model
issuing 3,000 kernel launches per forward pass at 6 µs dispatch each, when GPU compute totals (a) 10 ms and
(b) 120 ms. In which regime is the model framework-overhead-bound, and what is the most effective remedy without
writing new code?

**Answer.** Dispatch costs 3,000 × 6 µs = 18 ms. (a) The step takes about max(18, 10) = 18 ms and the GPU is busy
about 10 / 18 = 56% of it: framework-overhead-bound. (b) The step takes about max(18, 120) = 120 ms and the
dispatch is hidden behind the GPU queue: GPU-bound. In eager mode, raise the batch size until the kernels hide the
dispatch; when batching is not possible, compile the model with `torch.compile` (Chapter 4), which fuses the
operations and removes the per-operation dispatch.

**Explanation.** Because the CPU and GPU overlap, the step time is roughly the larger of total dispatch time and
total GPU time, and their sum is only a no-overlap upper bound (*The anatomy of an eager execution step*,
Figure 3.6). The chapter names raising the batch size as often the most effective step in eager mode, and fusion
by `torch.compile` as the fix when the batch cannot grow.

## Question 5

Explain why the DataLoader uses worker processes when threads are cheaper to start. What does this imply for
(a) host RAM when the dataset holds a large in-memory object, and (b) the size of per-sample return values?

**Answer.** Loading and preprocessing are CPU-bound Python, and the GIL lets only one thread run Python bytecode at
a time, so threads would run one after another; separate processes each have their own interpreter and GIL and run
in parallel across cores. (a) Each worker tends to end up with its own copy of a large structure of Python objects:
with fork, reference-count updates gradually copy the copy-on-write pages, and with spawn or forkserver the dataset
is pickled into each worker, so host RAM use multiplies with the worker count. (b) Workers return results by
pickling them between processes, so a large Python object per sample makes the hand-off itself the bottleneck.

**Explanation.** These consequences follow from the process-based design described in *Why workers are processes*;
the same section notes that starting workers has a cost, which `persistent_workers=True` amortizes across epochs.

## Question 6

Identify the performance anti-patterns in this inference code and the minimal set of fixes:
`for query in queries: tokens = tokenizer(query, return_tensors='pt').to('cuda'); output = model(**tokens);
score = output.logits.max().item(); results.append(score)`.

**Answer.** Four anti-patterns: autograd is not disabled; the loop runs one forward pass per query; each iteration
copies inputs to the GPU separately; and `.item()` forces a device synchronization and a device-to-host copy on
every iteration. Minimal fix: tokenize the queries as one padded batch, move it to the GPU once, run one forward
pass inside `torch.inference_mode()`, compute the maxima on the GPU (for example `logits.amax(dim=(1, 2))`, or over
the non-padded positions), and copy the result to the host once with `.tolist()` after the loop.

**Explanation.** A Python loop over samples pays for N kernel launches, N allocations and N graph builds (3.9× slower
than the batched call for BGE-M3 on a T4); reading a scalar off a GPU tensor blocks the host until the queue drains;
and one large transfer costs less than many small ones (*Common performance anti-patterns and what not to
over-engineer*, *Avoiding autograd at inference*).

## Question 7

In the Chapter 2 baseline on the A100, decode takes 336 ms of 379 ms (Table 3.11). Using Amdahl's Law, compute the
largest end-to-end speedup you could get by making every non-decode stage take zero time. What does the result
imply about where to focus next?

**Answer.** The non-decode stages are 11% of the A100 request, so the largest speedup is 378.9 / 336.4, about
1.13×. Work outside decode cannot reach the 2× target; the next work has to reduce decode time.

**Explanation.** Table 3.11 gives the same bound on the H100 (1.13×) and 1.32× on the T4, where decode's share is
smaller; decode owns 89% of the A100 request, and its per-step overhead is the target of Chapter 4 (*Where the time
goes after Chapter 3*).
