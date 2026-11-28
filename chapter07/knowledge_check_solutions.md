# Chapter 7 Knowledge check: solutions

Scaling Out: Choosing a Distributed Training Strategy

Section names refer to the chapter.

## Question 1

List the four occupants of training memory, give the per-parameter byte cost of the three model-state occupants
under mixed-precision Adam, and explain which occupants inference does not pay.

**Answer.** Parameters (2 bytes per parameter in 16 bits), gradients (2 bytes), optimizer states (12 bytes: FP32
master weights and the two Adam moments), and activations, which scale with batch size and sequence length. The
model state totals about 16 bytes per parameter. Inference pays for the 16-bit parameters plus transient
activations and, for language models, the KV cache; it pays for no gradients and no optimizer states.

**Explanation.** For a 7-billion-parameter model this is about 130 GB per replica for training against 14 GB of
weights for serving (*Why a Single GPU Runs Out*, Figures 7.1 and 7.3, Principle #19).

## Question 2

A 13-billion-parameter model does not fit on an 80 GB GPU for training, and a colleague proposes data parallelism.
Compute the model-state memory, identify the wall, and explain why data parallelism is the wrong fix.

**Answer.** 13 × 10⁹ × 16 bytes = 208 GB of model state before activations: the memory wall. Data parallelism
replicates the full model state on every GPU, so each replica still needs 208 GB. Sharded data parallelism
(ZeRO/FSDP) is the first option to try; with ZeRO-3 over P GPUs the model state falls to about 16/P bytes per
parameter, 26 GB per GPU on 8 GPUs.

**Explanation.** The memory wall is a problem of where to put state and the compute wall a problem of throughput,
and data parallelism does nothing for the memory wall (*Why a Single GPU Runs Out*, Table 7.1, Table 7.2,
Principle #21).

## Question 3

State the per-GPU communication cost of a ring all-reduce over P GPUs and explain why it is nearly independent of P.
Then explain the sense in which "independent of P" is an overstatement, and when that matters.

**Answer.** Each GPU sends 2N(P − 1)/P bytes for an N-byte buffer, which approaches 2N as P grows. The bandwidth term
stops growing, but the ring takes 2(P − 1) steps, each with a message-startup latency, so the latency term grows
with P. It matters for small payloads and very large P, where NCCL switches to double binary trees, whose latency
grows with log P.

**Explanation.** The cost model is 2(P − 1)·α + 2(P − 1)/P · N/B (*The ring all-reduce and its cost*,
Principle #20).

## Question 4

Contrast the parameter-server and ring topologies for gradient averaging. Why does the parameter server's cost grow
with the number of workers while the ring's does not?

**Answer.** A parameter server receives gradients from all P − 1 workers and sends the result back to each, so the
traffic through that one device is proportional to 2N(P − 1) and grows with P. In a ring each GPU sends only to its
neighbor and the buffer is split into P chunks, so each GPU moves 2N(P − 1)/P bytes, close to 2N for any P.

**Explanation.** The central node is the bottleneck in the first design, and the ring has no central node
(*The ring all-reduce and its cost*, Figure 7.7).

## Question 5

Explain why DistributedDataParallel is preferred over the older DataParallel, and name the two mechanisms in
DataParallel that limit its performance.

**Answer.** DDP runs one process per GPU, and the processes synchronize only through a bucketed gradient all-reduce
that overlaps with the backward pass. DataParallel drives all GPUs from one Python process with threads, so
(1) the global interpreter lock serializes the threads, and (2) it gathers outputs and gradients onto one device,
recreating the parameter-server bottleneck.

**Explanation.** PyTorch's documentation recommends DDP over DataParallel even on a single machine (*From
DataParallel to DistributedDataParallel*).

## Question 6

Data parallelism changes the effective batch size. State how, why that can hurt convergence, the standard
mitigation, and when even that is not enough.

**Answer.** With P replicas and a local batch b, the update uses a global batch of P × b. Past a point, larger
batches waste data and can converge to worse solutions or fail to converge at the original learning rate. The
mitigation is the linear scaling rule (scale the learning rate with the batch) with a warmup period. For the very
large batches of thousand-GPU jobs, layer-wise adaptive optimizers such as LARS and LAMB are needed.

**Explanation.** Scaling efficiency and training correctness are separate measures (*Effective batch size and the
convergence caveat*).

## Question 7

Describe what each ZeRO stage shards and the approximate per-GPU model-state memory under P-way sharding. Which
stage adds communication beyond plain data parallelism, and how much?

**Answer.** ZeRO-1 shards the optimizer states: 4 + 12/P bytes per parameter. ZeRO-2 also shards the gradients:
2 + 14/P. ZeRO-3 also shards the parameters: 16/P. ZeRO-1 and ZeRO-2 move the same volume as data parallelism;
ZeRO-3 adds the parameter all-gathers, about 50% more communication (about 3N bytes per GPU per step against 2N).

**Explanation.** A reduce-scatter plus an all-gather move the same bytes as one all-reduce, and ZeRO-3 gathers each
layer's parameters again in the forward and backward passes (*The three stages of ZeRO*, Table 7.2, *What the
sharding costs in communication*).

## Question 8

Tensor parallelism must run on intra-node NVLink, while pipeline parallelism can span the inter-node network.
Explain the placement rule in terms of what each communicates and how often.

**Answer.** Tensor parallelism runs two all-reduces per layer in the forward pass and two in the backward pass, for
every layer, every microbatch and every step, on the critical path: about 7.5 GB per microbatch per GPU for a
7-billion-parameter model with s = 4,096, b = 1 and t = 8. Pipeline parallelism sends only the activations at the
P − 1 stage boundaries, point to point, once per boundary per microbatch.

**Explanation.** Match each dimension's communication frequency to the interconnect that can carry it
(*Tensor parallelism: splitting a layer across devices*, Table 7.4, Principle #22).

## Question 9

For a pipeline with P stages and M microbatches, write the bubble fraction and explain how to make it small. Why
does 1F1B not reduce the bubble, and what does it reduce instead?

**Answer.** The bubble is (P − 1)/(M + P − 1) of the total time, or (P − 1)/M on top of the ideal compute time. Use
many microbatches relative to the number of stages, or the interleaved schedule, which reduces it to
(P − 1)/(v·M) with v chunks per GPU. 1F1B changes the order of forward and backward passes and keeps the same
startup and drain, so the bubble is unchanged; it bounds the in-flight activations to about the number of stages,
which cuts peak activation memory.

**Explanation.** *Pipeline parallelism: splitting layers into stages*, Figures 7.15 and 7.16.

## Question 10

In 4D parallelism, tensor parallelism is innermost and data parallelism outermost. Justify the ordering using
communication frequency and the interconnect hierarchy.

**Answer.** Tensor parallelism communicates every layer and needs NVLink, so it stays within a node. Pipeline
parallelism hands off activations at stage boundaries and tolerates the inter-node network, so it spans node
groups. Data parallelism all-reduces the gradients once per step, the communication most tolerant of network
latency, so it replicates the tensor-and-pipeline unit across the cluster.

**Explanation.** The composition follows from the placement rules of each strategy (*Hybrid and 4D Parallelism*,
Figure 7.17).

## Question 11

A Mixture-of-Experts layer routes each token to 2 of 64 experts. Explain how this decouples capacity from
per-token compute, and name the two all-to-all collectives and what each does.

**Answer.** All 64 experts add parameters, so capacity grows with the expert count, while each token runs through
only 2 of them, so the computation per token stays roughly fixed. Expert parallelism places experts on different
GPUs and needs a dispatch all-to-all that sends each token to the GPU holding its chosen expert, and a combine
all-to-all that returns the expert outputs to the token's original GPU.

**Explanation.** *Mixture-of-Experts Parallelism*, Figure 7.20.

## Question 12

You measure 95% scaling efficiency on a 64-GPU run. What measurement must accompany that number, and what failure
does scaling efficiency alone hide?

**Answer.** The time to reach a target validation metric, compared with the single-GPU or smaller baseline. Scaling
efficiency alone hides a convergence failure: the enlarged global batch can change the optimization so that the
model is worse while the GPUs stay busy.

**Explanation.** Scaling efficiency measures whether you are using the hardware, and only time to accuracy measures
whether you are getting the result; report both (*Benchmarking Distributed Training*, Figure 7.24).
