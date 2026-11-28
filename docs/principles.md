# The 51 Performance Engineering Principles

The book states 51 Performance Engineering Principles in boxes, numbered in order. Each one summarizes a rule that a chapter derives and then tests on the case study. The list gives each principle's number, its title as printed, one sentence from its box, and the chapter that states it. The box in the chapter gives the full statement and the measurement behind it.

## Chapter 1: The Imperative of Performance in Modern AI

Principles #1 to #2. Chapter page: [Chapter 1](book/chapter01.md).

| # | Principle | Statement |
|---|---|---|
| 1 | Measure before you optimize | Before changing anything, profile the system end-to-end to identify the actual bottleneck. |
| 2 | Define targets first | Before optimizing anything, write down the performance specification (p95 latency, minimum throughput, cost and reliability targets), and stop optimizing when the numbers are met. |

## Chapter 2: A Methodology for Performance Engineering

Principles #3 to #5. Chapter page: [Chapter 2](book/chapter02.md).

| # | Principle | Statement |
|---|---|---|
| 3 | Profile end-to-end before drilling down | Begin every investigation with an end-to-end profile. |
| 4 | Match the optimization to the bottleneck | Compute-bound code benefits from kernel fusion, mixed precision and Tensor Cores; memory-bandwidth-bound code from smaller data types, fused operations and on-chip blocking; communication-bound code from gradient compression, overlap with computation and topology-aware collectives. |
| 5 | Lock in gains with regression tests | Every accepted performance optimization should be paired with a regression test that fails if the optimization is later undone. |

## Chapter 3: PyTorch Internals and Execution Performance

Principles #6 to #7. Chapter page: [Chapter 3](book/chapter03.md).

| # | Principle | Statement |
|---|---|---|
| 6 | Disable autograd at inference | Wrap inference in torch.no_grad() or torch.inference_mode(). |
| 7 | The 10% rule for custom optimization | Before you write a custom optimization (a kernel, a data loader, a memory pool), check that the operation contributes at least 10% of step time and that no library function or configuration change gets you the same win. |

## Chapter 4: Optimizing with torch.compile and Kernel Fusion

Principles #8 to #10. Chapter page: [Chapter 4](book/chapter04.md).

| # | Principle | Statement |
|---|---|---|
| 8 | Compile the whole forward, then remove the breaks | Start by compiling with fullgraph=True and fix what it raises until the forward pass is a single graph. |
| 9 | Fusion cuts memory traffic | Fusion saves memory traffic, so it helps in proportion to how memory-bound the code is. |
| 10 | Match the compile mode to the workload | Use the default mode first, reduce-overhead when the work is launch-bound (small batches and small models, as in autoregressive decode), and max-autotune when you will run the compiled model enough to recover a long compile search. |

## Chapter 5: GPU Architecture and Hardware-Aware Decisions

Principles #11 to #14. Chapter page: [Chapter 5](book/chapter05.md).

| # | Principle | Statement |
|---|---|---|
| 11 | Saturate before you optimize | Before tuning any kernel, confirm it has enough parallel work to keep the GPU's SMs busy. |
| 12 | Measure bandwidth against the sustained copy rate | Measure achieved memory bandwidth against the sustained bandwidth of a large device-to-device copy on the same GPU, because sustained bandwidth sits below the datasheet value. |
| 13 | Classify first, optimize second | Determine whether your kernel is memory-bound or compute-bound before choosing an optimization strategy. |
| 14 | Tensor Cores are conditional | On the T4, the 4× throughput advantage of Tensor Cores over FP16 CUDA cores (8× over FP32) is available only for matrix-multiply-shaped operations with compatible shapes and aligned data. |

## Chapter 6: Practical High-Performance Kernels with Triton

Principles #15 to #18. Chapter page: [Chapter 6](book/chapter06.md).

| # | Principle | Statement |
|---|---|---|
| 15 | Fuse to cut memory traffic | A custom kernel wins on a memory-bound operation by loading inputs once, doing all the work on-chip, and writing outputs once. |
| 16 | Use the vendor library for plain matrix multiplication | For plain matrix multiplication, use the vendor library; write a matmul in Triton to learn the machinery or to fuse something onto it that the library cannot express. |
| 17 | Profile first, then fuse the bottleneck | For an operation that a library already implements, write a custom kernel only after profiling confirms three things: the operation is on the critical path, it is memory-bound, and its neighbors are fusable into it. |
| 18 | A kernel is not done until it matches a reference | Validate every custom kernel against a plain PyTorch implementation with torch.testing.assert_close at a dtype-appropriate tolerance, and test the awkward shapes, meaning non-power-of-two, prime, and non-multiples of the block, because those exercise the masks where correctness bugs hide. |

## Chapter 7: Scaling Out: Choosing a Distributed Training Strategy

Principles #19 to #23. Chapter page: [Chapter 7](book/chapter07.md).

| # | Principle | Statement |
|---|---|---|
| 19 | Budget training memory before choosing a strategy | Mixed-precision Adam training costs about 16 bytes per parameter for model state alone: 2 for the 16-bit (bf16 or fp16) weights, 2 for the 16-bit gradients, and 12 for the fp32 optimizer states (master weights, first moment, second moment). |
| 20 | Estimate communication time from the bytes each collective moves | The cost that determines whether a distributed strategy scales is the volume and frequency of communication. |
| 21 | Shard redundant state before you split the model | When a model overflows one GPU, sharding the optimizer states, gradients, and parameters across data-parallel replicas (ZeRO / FSDP) is the first option to try, because it keeps the simple data-parallel programming model and its communication cost is modest: free for optimizer-state and gradient sharding, about 50 percent more for parameter sharding. |
| 22 | Put the most communication-heavy parallelism on the fastest interconnect | Tensor parallelism communicates on every layer of every step, on the critical path, so it belongs on the highest-bandwidth interconnect available, which is intra-node NVLink. |
| 23 | Among strategies that fit, choose by step time and exposed communication | Choose the simplest strategy that makes the model fit, then verify that its communication overlaps with computation on your actual interconnect. |

## Chapter 8: High-Performance Inference and Serving

Principles #24 to #30. Chapter page: [Chapter 8](book/chapter08.md).

| # | Principle | Statement |
|---|---|---|
| 24 | Report serving throughput together with the latency at which it was measured | Measure goodput, the requests per second that meet both a TTFT and a TPOT objective, and report latency as percentiles, with the mean only next to them. |
| 25 | Schedule at the iteration level and limit each iteration by a token budget | Admit and retire requests at every forward pass, cap each pass with a token budget, and split long prefills into chunks that fit it. |
| 26 | Size a serving deployment by its KV cache capacity | Compute the KV bytes per token from the model configuration, multiply by the context you expect, and divide the memory left after the weights to get the number of requests the GPU can hold. |
| 27 | Allocate the KV cache in fixed-size blocks and reuse cached prefixes | Fixed-size blocks waste at most one partial block per request and allow sharing; keep finished prefixes, order prompts so that the stable part comes first, and move cold caches to cheaper memory. |
| 28 | Compare the memory traffic of an operator with the minimum its inputs and outputs require | Before optimizing arithmetic, compare the traffic an implementation generates with the minimum its inputs and outputs require. |
| 29 | Use speculative decoding only when decode is memory-bound | Speculative decoding helps when decode is memory-bound, at small batch or long context, and costs throughput when the batch already saturates compute. |
| 30 | Plan MoE memory by total parameters and MoE compute by active parameters | An MoE model must hold all experts but computes with few, so each expert sees a small share of every batch. |

## Chapter 9: Model Compression: Quantization and Sparsity

Principles #31 to #36. Chapter page: [Chapter 9](book/chapter09.md).

| # | Principle | Statement |
|---|---|---|
| 31 | Compress the tensor that limits the workload | Weights limit decode at small batch, activations and their matrix multiplications limit prefill and large batches, and the KV cache limits concurrency at long context. |
| 32 | Give each scale a group of values with a similar range | The error of linear quantization is set by the step, and the step is set by the largest value that shares the scale. |
| 33 | Measure how far the quantized model's outputs move from the original, on your own data | Report the KL divergence or top-1 agreement against the original model on your prompts, task accuracy with paired error bars and flips, and accuracy on the tasks the model serves. |
| 34 | Use the lowest-precision format that meets your quality target and that the hardware executes natively for the phase that limits the workload | Weight-only formats speed up memory-bound decode on any GPU that can convert them in the kernel; weight and activation formats speed up compute-bound prefill and large batches only where the Tensor Cores support them. |
| 35 | Quantize the KV cache when its bytes limit concurrency or long-context decode, and test it at the longest context you serve | The gain is largest for models with many KV heads and long requests and disappears when the scheduler limit binds first. |
| 36 | Prune in a pattern that your hardware and serving software execute faster, and measure the speedup end to end | Unstructured zeros give little speedup on a GPU unless the sparsity is very high, 2:4 sparsity needs kernel support in the serving engine and recovery training to reach acceptable accuracy, and structured pruning gives a smaller dense model that runs faster without special kernels. |

## Chapter 10: Efficient Model Adaptation and Maintenance

Principles #37 to #41. Chapter page: [Chapter 10](book/chapter10.md).

| # | Principle | Statement |
|---|---|---|
| 37 | Measure what an update makes the model forget as well as what it learns | A release gate for an updated model needs a fixed suite of tasks outside the target domain, evaluated as a paired comparison with the previous model, with paired error bars. |
| 38 | Choose the adaptation method from the memory budget, and measure QLoRA's throughput on your GPU and library version before assuming a cost | Full fine-tuning needs about 16 bytes per parameter, LoRA about 2, and QLoRA about 0.5 for the linear layers, with activations on top of each. |
| 39 | Merge an adapter when one adapter serves the traffic, and keep adapters separate when many share one base model | A merged adapter has the inference cost of the base model. |
| 40 | Distill when a teacher already exists and the student will serve many requests | Distillation transfers a large model's behavior to a model that is cheaper to serve. |
| 41 | Put new facts in the retrieval index and new behaviors in the weights | Updating an index takes minutes and leaves the model unchanged, and it outperformed fine-tuning for injecting new facts in the studies cited here. |

## Chapter 11: Navigating the Accelerator and Compiler Landscape

Principles #42 to #46. Chapter page: [Chapter 11](book/chapter11.md).

| # | Principle | Statement |
|---|---|---|
| 42 | Compare accelerators on the roof that bounds your workload | Decode at small batch is bound by the bandwidth of the memory that holds the weights and KV cache, prefill and training by dense compute in the format you will deploy, and feasibility by capacity. |
| 43 | Treat the software stack as part of the hardware specification | Report the fraction of peak that your model reaches with the compiler, kernels and serving engine you will run in production, next to the datasheet peak. |
| 44 | Treat determinism as a requirement with a measured cost | Decide whether a workload needs run-to-run, batch-invariant or cross-platform reproducibility, enable only the controls that level requires, and measure their throughput cost. |
| 45 | Keep a portable reference path, and specialize where profiling shows the time is spent | Maintain an exported, framework-neutral version of the model, such as a torch.export program, StableHLO or ONNX, as the baseline for correctness and portability. |
| 46 | Establish numerical parity before measuring speed on a new platform | Check agreement at the operator, layer and task level against a fixed reference, with tolerances set by the number format, and accept throughput numbers only after the model passes. |

## Chapter 12: The Mature ML System

Principles #47 to #51. Chapter page: [Chapter 12](book/chapter12.md).

| # | Principle | Statement |
|---|---|---|
| 47 | Account cost per completed task at the latency objective | Divide the infrastructure cost by the tasks completed within the objective, and include the idle headroom, CPU, network and storage that a cost per GPU-hour leaves out. |
| 48 | Size prefill capacity for the TTFT objective and decode capacity for the TPOT objective | Estimate the GPUs each phase needs from the workload's prompt and output lengths, and when one pool cannot meet both objectives, give each phase its own pool and parallelism and budget for moving the KV cache between them. |
| 49 | Measure multi-step workloads by task completion time | An agent's latency is the sum of its steps, and most of each step's prompt was already processed in the previous step. |
| 50 | Gate every change on pinned measurements of latency, throughput and accuracy | Run the same workload on the same container digest and GPU type, compare with a stored baseline using thresholds larger than the run-to-run variation, and block a release that degrades any objective. |
| 51 | Evaluate an architecture by its measured cost on the target hardware and serving workload | KV bytes per token, active parameters and the number of sequential steps per token set the cost of inference together with the parameter count. |
