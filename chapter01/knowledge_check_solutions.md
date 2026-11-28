# Chapter 1 Knowledge check: solutions

The Imperative of Performance in Modern AI

The same answers appear at the end of the Chapter 1 lab notebook (`chapter01_lab.ipynb`). Section names refer to
the chapter.

## Question 1

A fraud detection model achieves 98.2% accuracy but requires 500 ms per inference, and the SLA requires sub-50 ms
response time. (a) How would you frame this as a performance engineering problem? (b) What are the first two
diagnostic questions you would ask before attempting any optimization, and why?

**Answer.** (a) The model meets the accuracy requirement and misses the latency requirement by a factor of ten, so
the task is to reach the 50 ms target at the same accuracy. (b) First, where do the 500 ms go, stage by stage?
Second, what is the full specification: the percentile and load at which 50 ms must hold, and the throughput,
cost and reliability targets?

**Explanation.** The chapter calls this distance between what a model achieves in evaluation and what it can
deliver under production constraints the accuracy–performance gap, and Table 1.1 lists real-time fraud detection
with a critical latency priority (*Translating lab accuracy to production value*, *The accuracy–performance gap*).
The two questions are Principle #1, measure before you optimize, and Principle #2, define targets first
(*Business constraints and real-world trade-offs*): without the stage breakdown the work may go to a stage that is
not the bottleneck, and without the specification there is no stopping point.

## Question 2

Explain the difference between latency and throughput in an LLM serving system. Describe a scenario where
optimizing for throughput increases latency, and the mechanism.

**Answer.** Latency is the time of one request, from arrival to the last byte of the response, and is described by
its distribution (p50, p95, p99). Throughput is the work the whole system completes per unit time, in requests or
tokens per second. Batching decode steps raises throughput and, past a GPU-dependent batch size, lengthens every
request's decode steps.

**Explanation.** On the T4, going from batch size 1 to 64 raised Gemma-2B-IT decode throughput from 38 to 1,050
tokens/s while each decode step grew from 26 to 61 ms, so every request in the batch waited longer (*Throughput*,
Figure 1.4). With static batching a request also waits until its batch is formed, and each step slows once the
batch exceeds what the GPU processes at a constant step time (*The interdependence of the four factors*).

## Question 3

Rank the four performance factors for a production chatbot serving 10,000 concurrent users, then for a nightly
batch pipeline that summarizes 1 million documents. How does the ranking change, and why?

**Answer.** Chatbot: latency and reliability first, then throughput and cost. Nightly batch: throughput and cost
first, then reliability, with latency last.

**Explanation.** Table 1.1 gives the interactive chatbot a high latency priority (under 2 s TTFT), high reliability
and moderate throughput and cost, because users see each request; it gives batch document work critical throughput
and cost priorities and a low latency priority, because only the completion of the whole job matters
(*The interdependence of the four factors*). The ranking follows from who waits for the result: a user waiting on
one request, or a schedule waiting on the whole batch.

## Question 4

The LLM decode stage dominates naive end-to-end latency (Table 1.3). Explain why memory bandwidth sets the lower
bound on decode step time, why the measured A100 step is several times slower than that bound, and what this
implies about the bottleneck.

**Answer.** Each decode step reads all model weights from GPU memory to produce one token, so the step cannot be
faster than the weight bytes divided by the memory bandwidth: 5.0 GB of FP16 weights at the A100's peak bandwidth
take about 2.5 ms. The measured step takes about 17.7 ms in the lab's decode loop (16.5 ms with `generate()`),
because per-step host overhead adds most of the time. Decode is the stage to work on first, and on the A100 the
first target is that overhead.

**Explanation.** Decode at batch 1 performs about one floating-point operation per byte of weights read and ran at
about 15% of its bandwidth ceiling on the A100 (*The GenAI inference problem: Stateful and memory-hungry*,
Figure 1.12). Decode takes 89% of the sum of stage medians on the A100 and 76% to 88% on the H100 and T4, so the
generation stage comes before the retrieval engine and the embedding model (*Latency budget of the naive
pipeline*, Table 1.3); Chapter 2 attributes the rest of each step to host overhead.

## Question 5

Training GPT-4 is reported to have cost more than $100 million; take $100 million. At 10 million queries per day
and $0.01 per query, how long before inference costs exceed training costs? What does this imply about engineering
investment?

**Answer.** Inference costs 10,000,000 × $0.01 = $100,000 per day, so it passes $100 million after 1,000 days,
about 2.7 years. Engineering effort on inference efficiency returns more than the same effort on training
efficiency for a model that stays in production.

**Explanation.** The chapter reports inference shares of 60% to 90% of ML compute in the cited reports and states
that a 2× improvement in inference efficiency is worth more than a 2× improvement in training efficiency, because
the inference saving recurs every day the model is in production and the training saving occurs once (*Cost*,
Figure 1.7). This is why most of the book's performance work targets serving (*The accuracy–performance gap*).

## Question 6

Identify two differences in compute characteristics that make GPU kernel optimization for ML workloads (for
example, attention) different from kernel optimization for traditional HPC workloads (for example, finite-element
simulation). Consider memory access patterns, data types and parallelism structure.

**Answer.** Any two of the following. Data types: HPC codes have traditionally used FP64 for solver stability,
while ML training tolerates BF16 and accelerators deliver an order of magnitude more throughput in it (about 15× on
the H100, dense BF16 against FP64 Tensor Core peaks). Memory access: HPC kernels work on structured, regular grids
with predictable access, while attention is irregular and data-dependent and decode updates a growing KV cache at
every step. Parallelism: HPML adds tensor parallelism, which splits weight matrices, and pipeline parallelism,
which splits layers, to data parallelism.

**Explanation.** The irregularity makes classical HPC optimizations such as stencil computation and cache-oblivious
layouts only partly applicable, and the lower precision moves the limit of many inference kernels to memory
bandwidth, which grew about 21× from the V100 to Rubin against about 280× for peak Tensor Core throughput
(*HPML versus traditional HPC*, Figure 1.9).
