# Chapter 2 Knowledge check: solutions

A Methodology for Performance Engineering

## Question 1

A team reports their inference system "does 5,000 RPS." What extra information do you need to interpret that number, and why?

**Solution.** A single throughput number is meaningless on its own, because the same engineering team can report wildly different RPS from the same hardware depending on the operating point. As the chapter puts it, "Our system does 10,000 RPS" is meaningless without the context that pins it down. To interpret 5,000 RPS you need at least the following properties.

- Latency at which the throughput is sustained, reported as a percentile, not a mean. Throughput and latency are coupled by queueing (Little's Law: average concurrency equals throughput times average response time), so RPS only means something paired with the p95 or p99 it holds. A system that holds 200 ms p99 at 400 RPS can degrade to a 2-second p99 at 480 RPS while the mean latency barely moves. 5,000 RPS at a 5-second p95 (batch summarization) is a different system from 5,000 RPS at a 100 ms p95 (search ranking).
- Batch size. The throughput-latency Pareto frontier is traced out by varying batch size, so the batch size is part of the operating point being quoted.
- Input distribution, including sequence length. Generative throughput depends heavily on prompt length and output length, so RPS without the input profile cannot be compared against another system.
- Hardware identity. GPU model, interconnect, and memory bandwidth determine the achievable ceiling. 5,000 RPS on one accelerator is not 5,000 RPS on another.

Other valid additions exist: precision (FP16, INT8, FP4), the serving framework and its version, and whether the number was measured at the load the team intends to serve rather than on an optimistic microbenchmark. The underlying point from the chapter holds: throughput must be reported jointly with latency, batch size, input distribution, and hardware, or it is marketing rather than measurement.

## Question 2

The arithmetic mean of two throughput measurements is generally the wrong summary. Which mean is correct, and why?

**Solution.** Throughput is a rate (requests or tokens per unit time), and the correct summary for rates, when total time is what matters, is the harmonic mean. The harmonic mean is the reciprocal of the arithmetic mean of the reciprocals, which weights each rate by the time spent at it. The arithmetic mean of rates silently assumes equal time at each rate, which is almost never the operating condition you care about.

Concrete example. Suppose a system processes an equal amount of work in two phases, 600 requests in each phase. In phase one it runs at 100 RPS; in phase two it runs at 300 RPS.

```
phase 1: 600 requests at 100 RPS -> 6.0 s
phase 2: 600 requests at 300 RPS -> 2.0 s

aggregate throughput = total requests / total time
                     = 1200 / (6.0 + 2.0)
                     = 1200 / 8.0
                     = 150 RPS

arithmetic mean of rates = (100 + 300) / 2 = 200 RPS   (wrong)
harmonic mean of rates   = 2 / (1/100 + 1/300) = 150 RPS   (correct)
```

The aggregate throughput a user actually observes over the whole run is 150 RPS, which the harmonic mean reproduces exactly and the arithmetic mean overstates by a third. The arithmetic figure of 200 RPS overweights the fast phase, which the system spends less time in. This is why the chapter assigns the harmonic mean to rates: it is the only mean that agrees with total-work-over-total-time.

(For completeness, and consistent with Table 2.1: the arithmetic mean is correct for additive quantities in the same units, such as wall time across trials, and the geometric mean is correct for ratios and normalized scores, such as speedups across a benchmark suite. Rates belong to the harmonic mean.)

## Question 3

Compute the ridge point for an H100, estimate the arithmetic intensity of a per-token forward pass over a 70B BF16 model, and explain why it must fall below the ridge.

**Solution.** The ridge point is the arithmetic intensity at which a workload transitions from memory-bandwidth-bound to compute-bound. It is the peak compute divided by the peak memory bandwidth. For the H100 SXM5 figures the chapter uses (BF16 dense peak approximately 989 TFLOP/s, HBM3 bandwidth approximately 3.35 TB/s):

```
AI_ridge = peak FLOPs / peak bandwidth
         = 989e12 FLOP/s / 3.35e12 byte/s
         ~= 295 FLOP/byte
```

Any workload whose arithmetic intensity is below 295 FLOP/byte cannot saturate the H100's tensor cores even in principle, because its performance is pinned to the sloped (bandwidth-limited) section of the roofline.

Now the per-token decode step for a 70B-parameter model in BF16. At decode the batch is effectively one token, so the dominant cost is streaming every weight once from HBM to perform a single matrix-vector product per layer. The two quantities are:

```
bytes moved  ~= 2 bytes/parameter x 70e9 parameters = 1.4e11 bytes
              (BF16 is 2 bytes, and every weight is read once per token)

FLOPs        ~= 2 FLOP/parameter x 70e9 parameters   = 1.4e11 FLOP
              (a multiply-add per weight is 2 FLOPs, dense inference ~= 2N)

AI = FLOPs / bytes = 1.4e11 / 1.4e11 ~= 1.0 FLOP/byte
```

The arithmetic intensity is on the order of 1 FLOP/byte. This is not an accident of the parameter count: the 2N FLOPs and the 2N bytes both scale linearly with N, so N cancels and the ratio is roughly 1 FLOP/byte for any dense model read once per generated token in BF16. The lower precision only moves the bytes-per-parameter term (INT8 would read 1 byte per parameter and push the ratio toward 2), it does not change the order of magnitude.

Since 1 FLOP/byte is roughly 295 times below the 295 FLOP/byte ridge, per-token decode sits far out on the memory-bound slope of the roofline. No kernel optimization can lift it off that slope, because the diagonal is a floor as well as a ceiling: once the algorithm fixes the bytes-per-FLOP ratio, achievable performance is capped at (peak bandwidth times arithmetic intensity). At 1 FLOP/byte over 3.35 TB/s that is roughly 3.35 TFLOP/s against a 989 TFLOP/s ceiling, so a single-token decode uses well under one percent of the H100's compute. Concretely, reading 1.4e11 bytes per token over 3.35 TB/s bounds throughput at roughly 24 tokens per second per sequence at batch size 1. The only levers that help are reducing bytes transferred (quantization, KV cache compression, distillation) or amortizing the transfer across more useful work (batching, speculative decoding). Faster kernels do not help.

## Question 4

A 64-GPU cluster delivers 32x speedup over a 1-GPU baseline. Use Amdahl's Law to estimate the implied serial fraction, and explain which training-step work is serial.

**Solution.** Amdahl's Law for the parallel regime, with serial fraction s = (1 - p) and N processors, is:

```
S(N) = 1 / ( s + (1 - s)/N )
```

Set N = 64 and S = 32 and solve for s:

```
1/S = s + (1 - s)/N

1/32 = s + (1 - s)/64

0.03125 = s + 0.015625 - 0.015625 s

0.03125 - 0.015625 = s (1 - 0.015625)

0.015625 = 0.984375 s

s = 0.015625 / 0.984375 ~= 0.0159  = about 1.6%
```

The implied serial fraction is approximately 1.6%. That small fraction is the optimization target, and it sets a hard ceiling: the maximum speedup is 1/s = about 63x no matter how many GPUs are added. Doubling the cluster to 128 GPUs yields only

```
S(128) = 1 / ( 0.0159 + 0.9841/128 ) ~= 42x
```

so going from 64 to 128 GPUs buys about 10x of additional speedup headroom at double the hardware, and the 64x ideal is unreachable without attacking the serial fraction itself. (The chapter's prose rounds this to "roughly 38x" at 128 GPUs; the exact arithmetic from a 1.6% serial fraction gives about 42x. Either way the lesson is identical: the return on added hardware collapses well before the ideal.)

Which training-step work is serial versus parallel:

- Serial (does not shrink as GPUs are added): the optimizer step that updates parameters and optimizer state, the gradient synchronization or all-reduce across devices (its payload is fixed by the model size, not by the device count, so it does not get cheaper per step with more GPUs and can grow with worse collective scaling), and the slowest stage of a pipeline-parallel schedule, which gates every other stage. These are the components that make up the roughly 1.6%.
- Parallel (shrinks as GPUs are added): the per-example forward and backward compute over the global batch, which data parallelism splits across devices. Adding GPUs at a fixed global batch size accelerates exactly this portion and nothing else.

So adding more GPUs accelerates the forward and backward passes but leaves the optimizer update, the parameter synchronization, and the slowest pipeline stage fixed. That is why strong-scaling efficiency degrades at extreme scale, and why beyond a point the engineering effort belongs on the serial fraction (overlapping communication with computation, reducing synchronization, balancing pipeline stages) rather than on buying more devices.

## Question 5

You profile a model and find 40% of CUDA time in aten::addmm. A colleague suggests a custom Triton kernel. Which two facts from this chapter's methodology would you establish first?

**Solution.** Two facts, both aimed at not repeating the opening vignette, where three weeks of expert kernel work moved end-to-end latency by 0.3%.

First, establish the bottleneck class of that matrix multiplication with the Roofline Model before writing any kernel. Profile it, classify it, then choose. If aten::addmm is memory-bandwidth-bound at the batch size and shapes in question (the decode case, where the GEMM is really a matrix-vector product streaming weights from HBM at an arithmetic intensity near 1 FLOP/byte), then it is already pinned to the bandwidth slope of the roofline and a faster kernel cannot help. The diagonal is a floor as well as a ceiling: no kernel can exceed (peak bandwidth times arithmetic intensity) once the algorithm fixes the bytes per FLOP. In that case the levers are quantization and KV cache compression and batching, not Triton. A Triton rewrite only pays off if the operator is genuinely compute-bound with headroom below the compute ceiling. This is Principle #4, match the optimization to the bottleneck, applied before committing engineering time.

Second, establish that this operator is actually on the critical path and that 40% of CUDA time is a large enough share of end-to-end wall-clock time to matter. CUDA time is not the same as wall-clock time: if the stage containing this GEMM is off the critical path, or if GPU compute is a small fraction of a request dominated by CPU, I/O, or framework overhead, then even eliminating the operator entirely moves end-to-end latency very little. Amdahl's Law bounds the payoff: if the addmm work is 40% of a GPU phase that is itself a modest share of the request, the achievable end-to-end speedup is small regardless of how good the kernel is. Confirm the share of total wall-clock time first, then decide whether the ceiling is worth three weeks.

If both facts favor the rewrite (the operator is compute-bound with headroom and it dominates the critical path), the Triton kernel is a reasonable investment. If either fails, it is wasted effort.

## Question 6

The chapter says re-profiling is mandatory after every accepted optimization. Give a concrete two-step scenario where skipping it causes the second optimization to fail.

**Solution.** The principle is that a successful optimization removes one constraint and exposes the next, so the profile that justified step one is stale for step two. Here is a concrete sequence on the chapter's RAG decode stage.

Step one. The initial end-to-end profile shows LLM decode dominating the request, and an Nsight Systems trace of the decode loop shows a repeating kernel-launch gap: a sawtooth of kernel, gap, kernel, gap on the HW stream, with the gaps matching the cudaLaunchKernel issue rate. That is the framework-overhead signature, the host cannot issue kernels fast enough. The team applies torch.compile and CUDA graphs to fuse operators and eliminate per-launch overhead. The gaps close, the GPU stream becomes densely packed, and decode gets meaningfully faster. The change is validated and accepted.

Step two, done wrong. Without re-profiling, the team reasons from the original trace, which still shows "framework overhead in decode" in their notes, and schedules more launch-overhead work: further fusion, more aggressive graph capture. They spend a week on it and ship essentially no gain.

Why it fails. After step one the bottleneck has shifted. With the launch gaps gone, decode is now pinned to the memory-bandwidth ceiling: the dominant kernels are large GEMMs streaming 2 bytes per parameter from HBM to produce one token, at an arithmetic intensity near 1 FLOP/byte and roughly 100% of the memory ceiling. No amount of additional fusion or graph capture changes the bytes moved per token, so the second optimization targets a constraint that no longer binds. A fresh 30-second profile after step one would have shown a dense, gap-free stream at the bandwidth ceiling and redirected the team to the levers that actually apply at that point: quantization to cut bytes per parameter, KV cache compression, and batching or speculative decoding to amortize the weight reads. Skipping the re-profile is invisible in the optimization-by-optimization view, because step one genuinely succeeded; the system simply stops getting faster because the team is no longer working on the real constraint.

## Question 7

Construct a regression test in pseudocode that catches a 15% latency regression in the RAG decode stage on a T4. State the warmup policy, iteration count, and threshold-with-margin, and discuss the tight-versus-loose threshold trade-off.

**Solution.** The test encodes the property we care about as an automated check with a bounded, statistically sound, hardware-pinned measurement, so a later refactor or library upgrade cannot quietly erode the gain (Principle #5). The design choices:

- Warmup policy: discard the first iterations entirely, since the first GPU iteration includes JIT compilation, kernel cache population, and CUDA context creation and runs 5 to 50 times slower than steady state. Use 10 warmup iterations, not recorded.
- Iteration count: 50 measured iterations after warmup, enough to estimate p95 with a stable noise floor while keeping the test well under the 10-minute CI budget.
- Timing: synchronize the device around the timed region (CUDA events, or torch.cuda.synchronize before start and before stop), or the measurement captures kernel queue time rather than execution time.
- Threshold with margin: pin the baseline p95 captured on the same physical T4, and set the fail threshold at 1.15 times that baseline so a true 15% regression trips it, with the recorded baseline itself taken with margin over run-to-run noise. The regression gate here is deliberately the 15% the question asks us to catch; the general chapter guidance of a 1.1 to 1.2x margin is the band this sits in.

```
# Regression test: RAG decode stage, T4 GPU
# Runs in CI on a hardware-pinned T4 on every pull request.

BASELINE_P95_MS = 4270.0      # decode-stage p95 locked in when the
                              # optimization was accepted, same T4
REGRESSION_FACTOR = 1.15      # fail on a 15% slowdown
THRESHOLD_MS = BASELINE_P95_MS * REGRESSION_FACTOR   # 4910.5 ms

WARMUP_ITERS = 10
MEASURE_ITERS = 50

def test_decode_latency_regression():
    model, inputs = load_rag_decode_fixture()   # fixed prompt, 20 output tokens, BS=1

    # Warmup: not recorded, absorbs JIT + context + cache population
    for _ in range(WARMUP_ITERS):
        run_decode(model, inputs)
    synchronize_device()

    # Measure: device-synchronized timing per iteration
    samples_ms = []
    for _ in range(MEASURE_ITERS):
        start = cuda_event(); stop = cuda_event()
        start.record()
        run_decode(model, inputs)
        stop.record()
        synchronize_device()
        samples_ms.append(start.elapsed_time(stop))

    p95 = percentile(samples_ms, 95)
    stddev = std(samples_ms)

    # Report full distribution, not just the mean
    log(f"decode p50={percentile(samples_ms,50):.1f}ms "
        f"p95={p95:.1f}ms sd={stddev:.1f}ms "
        f"threshold={THRESHOLD_MS:.1f}ms")

    assert p95 <= THRESHOLD_MS, (
        f"decode p95 {p95:.1f}ms exceeds {THRESHOLD_MS:.1f}ms "
        f"({REGRESSION_FACTOR:.2f}x baseline): latency regression")
```

Trade-off between a tight and a loose threshold. A tight threshold (margin close to 1.0, say 1.05) catches small regressions early, including slow drift that a loose gate would let accumulate across many pull requests, but it sits close to the measurement noise floor and produces false alarms when ordinary run-to-run variance pushes a clean build over the line. On a free-tier-style T4 with thermal throttling and noisy neighbors, that flapping erodes trust in the gate and trains developers to ignore or re-run it. A loose threshold (say 1.25) almost never flaps, so a failure is a real signal, but it lets genuine regressions up to that size ship unnoticed, and several sub-threshold regressions can stack into a large cumulative loss that no single pull request ever tripped. The resolution is to set the margin from the measured noise (flag only when the change exceeds the standard deviation band of the baseline), pin the hardware so machine-to-machine variance is removed, and report the full distribution so a human can see drift before it crosses the line. For a 15% target with a well-controlled T4, a 1.15 factor is the defensible middle: comfortably above typical noise, tight enough that a true 15% regression cannot hide.
