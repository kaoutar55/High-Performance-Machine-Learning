# Chapter 11 Knowledge check: solutions

Navigating the Accelerator and Compiler Landscape

Section names refer to the chapter.

## Question 1

An accelerator offers 2,300 TFLOP/s in BF16 and 7.4 TB/s of HBM bandwidth. Compute its BF16 ridge point and the
batch-size-1 decode bound for a model with 16 GB of BF16 weights. Which property would you change to double the
tokens per second at batch size 1?

**Answer.** Ridge point: 2,300 × 10¹² / 7.4 × 10¹² ≈ 311 FLOP/B. Decode bound: 16 GB / 7.4 TB/s ≈ 2.2 ms per token,
about 460 tokens/s. Batch-1 decode at about 1 FLOP/B is bound by memory bandwidth, so doubling the bandwidth (or
halving the bytes read per token) doubles the bound; more compute changes nothing.

**Explanation.** The ridge point is peak compute over bandwidth for the format in use, and decode at small batch is
bound by the bandwidth of the memory that holds the weights (*Five properties that describe an accelerator*,
Principle #42).

## Question 2

Why does a TPU program pad its inputs to a few fixed lengths, and what does the same technique achieve in a GPU
serving engine?

**Answer.** XLA compiles for static shapes, so every distinct prompt length would compile a new program; padding to
a small set of bucket sizes bounds the number of compiled programs. Serving engines use the same technique to bound
the number of CUDA graphs they capture, padding each step to the nearest captured batch size.

**Explanation.** *PyTorch beyond NVIDIA GPUs* (TorchTPU compiles with `dynamic=False`) and *What transfers between
stacks*; Chapter 8 describes the CUDA-graph padding.

## Question 3

An SRAM-based chip holds about 500 MB. How many chips does a 70B-parameter model in FP8 need for its weights alone,
and what else must fit in fast memory during long-context decode?

**Answer.** 70 × 10⁹ bytes / 500 MB ≈ 140 chips. The KV cache of every running request must also fit: at 128 KiB per
token for Llama-3.1-8B in BF16, a single 100K-token request needs about 13 GB, and it competes with the weights for
SRAM.

**Explanation.** *What limits dataflow designs*, Figure 11.6; one survey estimated about 576 first-generation Groq
chips for a 70B model.

## Question 4

The same prompt sent at temperature 0 returns two different completions at different times of day. Name the most
likely cause, and describe how you would confirm it.

**Answer.** Kernels that are not batch-invariant: a sequence's result depends on the batch it is processed with, and
under load the batch changes with every step. To confirm, send the same prompt many times while a background load
keeps different numbers of requests running, count the distinct completions and the first divergent token, and
repeat with batch-invariant kernels (`VLLM_BATCH_INVARIANT=1` in vLLM); the divergence should disappear.

**Explanation.** In the chapter's experiment (Llama-3.1-8B, vLLM 0.33.0), 100 identical requests gave one
completion alone on the GPU and up to 4 (A100) and 12 (H100) distinct completions with 64 requests running, and a
single completion in batch-invariant mode, at a throughput cost (*Determinism constraints*, *Measuring
nondeterminism on the case-study model*, Principle #44).

## Question 5

A model ported to a new accelerator matches the reference on every operator test but scores two points lower on its
task evaluation. List the steps you would take to find the cause.

**Answer.** Record and compare the full environment (drivers, framework, compiler, serving engine, model revision,
tokenizer, container digest, precision settings). Run a layer-level comparison with hooks that save every decoder
layer's output and look for a jump at one layer. Run model-level checks: KL divergence and top-1 agreement of the
next-token distributions against an FP32 reference, then greedy generations and task accuracy with paired error
bars. Check operator coverage and fallbacks, check precision defaults and the serving-stack configuration, and bisect
to a minimal reproduction.

**Explanation.** These are the steps of *Debugging Methodology for New Hardware* (Figure 11.9). Model-level checks
catch errors that operator tests miss: providers serving the same gpt-oss-120B weights scored between 80.0% and
93.3% on AIME 2025 because of their serving stacks (Principle #46).

## Question 6

Your workload is interactive decode for a 70B model with 32K-token contexts. Which properties from Figure 11.1 limit
it, and which evidence would you ask each vendor for?

**Answer.** Memory bandwidth of the memory that holds the weights and KV cache sets the per-token time; memory
capacity sets how many chips the weights and the 32K-token caches need; interconnect bandwidth matters once the
model is split across chips; and the software stack sets how close the workload comes to the bandwidth roof. Ask for
per-user tokens per second and TTFT, and capacity and TPOT, at 32K tokens of context, measured with the serving
stack you will deploy.

**Explanation.** Table 11.6 lists bandwidth for interactive decode and HBM capacity and scale-up bandwidth for long
context, with this evidence (*A decision table*, Principles #42 and #43).

## Question 7

An FP8 checkpoint runs on a GPU whose FP8 rate is twice its BF16 rate, and on an accelerator whose FP8 rate equals
its BF16 rate. On which phase would you expect FP8 to help on each, and why?

**Answer.** On the GPU, FP8 helps both phases: prefill, which is compute-bound, runs at twice the rate, and decode,
which is bandwidth-bound, reads half the weight bytes. On the accelerator (for example Gaudi 3, which runs FP8 at its
BF16 rate), FP8 helps decode through the halved bytes and gives prefill no compute gain.

**Explanation.** Compare accelerators on the roof that bounds each phase (Principle #42, *Number formats as a
hardware feature*). On the H100, FP8 weights cut Llama-3.1-8B's time per output token by 1.5× against a bound ratio
of 1.8, because the embedding and output head stay in BF16 (Table 11.10).

## Question 8

A JAX training step slows down after you change the sharding of one weight, and the Python code shows no new
communication. Where would you look, and what would you expect to find?

**Answer.** In the compiled program (`jit(...).lower(...).compile().as_text()`) or a profile: in JAX the shardings
propagate and the compiler inserts the collectives, so you would expect new all-gathers or reduce-scatters, or a
resharding of the changed weight, that the Python code does not show.

**Explanation.** *Sharding in JAX and PyTorch*; the communication costs of Chapter 7 apply unchanged.
