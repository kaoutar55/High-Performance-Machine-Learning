# Chapter 6 Knowledge check: solutions

Practical High-Performance Kernels with Triton

Section names refer to the chapter. Timings are the chapter's H100 measurements (PyTorch 2.14.1, Triton 3.8.0,
24 November 2026).

## Question 1

Explain the difference between CUDA's per-thread SIMT model and Triton's per-tile SPMD model. Name three
optimizations that are the programmer's responsibility in CUDA but the compiler's in Triton.

**Answer.** In CUDA you write the code of one scalar thread, compute its index and manage shared memory and
barriers yourself. In Triton each program instance operates on a whole tile with array operations, and the
compiler generates the per-thread code. Moved to the compiler: shared-memory allocation and staging, memory
coalescing, vectorization, thread-to-element mapping, software pipelining (`num_stages`) and Tensor Core
instruction selection (`tl.dot`); any three.

**Explanation.** Both models are SPMD over program instances; they differ in what one program operates on. The
programmer keeps the tile shapes, the work partition, pointer and stride arithmetic, boundary masks and the
accumulation dtype (*SPMD over tiles*, Table 6.1).

## Question 2

In the vector-addition kernel, what does `mask = offsets < n_elements` protect against, and what goes wrong if you
omit it on both the load and the store?

**Answer.** It protects the last program instance, whose tile runs past the end of the array when the length is
not a multiple of `BLOCK_SIZE`. Without it, the out-of-bounds loads read whatever lies past the array, or fault,
which corrupts the result; the out-of-bounds stores overwrite memory that may belong to another tensor.

**Explanation.** Mask errors are a frequent Triton bug and their symptoms rarely point to the mask (*Your First
Kernel: Vector Addition*, Table 6.5).

## Question 3

The fused softmax loads a masked row with `other=-float('inf')`. Explain why the choice of `other` depends on the
first reduction the kernel performs, and give an input for which `other=0` would produce a wrong result.

**Answer.** The padding lanes enter the first reduction, a row maximum, so their value must be the identity of
that reduction: a lane holding negative infinity does not win a max. With `other=0`, a row whose real values are all negative, for
example [-3.0, -1.0, -2.0] padded to four lanes, takes its maximum (0.0) from a padding lane, and every exponential
and sum after it is wrong.

**Explanation.** The rule generalizes: use negative infinity for a max and zero for a sum (*The fused softmax
kernel*, Table 6.5).

## Question 4

State the memory-traffic argument for why fusing a softmax is faster than eager PyTorch, and explain why the
speedup over `torch.nn.functional.softmax` is much smaller than the speedup over a naive implementation.

**Answer.** Eager PyTorch runs the max, subtraction, exponential, sum and division as separate kernels that each
read and write a full-size array; for a 4,096 × 4,096 FP16 input that is about 268 MB of traffic for 67 MB of
input and output. The fused kernel reads each row once and writes it once. `F.softmax` is already a fused
implementation, so the remaining gain comes from implementation details: on the H100 the Triton kernel was 3.90×
faster than the naive softmax and 1.47× faster than `F.softmax` at 4,096 × 4,096.

**Explanation.** For a memory-bound kernel time is approximately bytes divided by bandwidth, so cutting the bytes
cuts the time (*Fusion as the Main Optimization*); the speedup to report is the one measured against the fused
library (*Choosing the right baseline*, Figure 6.7).

## Question 5

The LayerNorm kernel reads the input row up to three times, yet the chapter calls it a "read once" fusion.
Reconcile the two statements using the memory hierarchy from Chapter 5.

**Answer.** The three passes (mean, variance, normalization) re-read the row from L1 or L2, or from registers in the
single-tile version, so the DRAM traffic stays at one read and one write per row while the row stays resident on
chip. DRAM traffic is the level that sets the runtime of this memory-bound kernel.

**Explanation.** The passes follow from numerical staging, since the variance needs the mean; Nsight Compute on the
target GPU confirms where the re-reads are served (*Fusing a Normalization Layer*).

## Question 6

Why does the matmul kernel allocate its accumulator in FP32 although inputs and output are FP16? What goes wrong
with FP16 accumulation, and on what kind of input does it show up first?

**Answer.** Each output element is a sum of K products; FP16 keeps 10 fraction bits, so once the running sum is
large each new product loses its low-order bits to rounding, and the error grows with K. It shows up first on large
reductions (large K, for example 4,096), where the result drifts from the reference.

**Explanation.** Accumulating in FP32 and casting only the final result keeps the error far below the FP16 storage
tolerance (*The accumulator and tl.dot*, Table 6.5).

## Question 7

The grouped program ordering in the matmul kernel changes no arithmetic yet raises throughput. Explain the
mechanism in terms of L2 reuse, and describe the access pattern of the naive row-major ordering.

**Answer.** Row-major ordering finishes a whole row of output tiles before the next row, sweeping through every
column tile of B; by the time the next row starts, the A and B tiles have been evicted from L2, so B is reloaded
from DRAM for every row of C. Grouped ordering processes output tiles in super-blocks `GROUP_SIZE_M` rows tall, so a
compact working set of A and B tiles stays in L2 and is reused across many output tiles.

**Explanation.** Triton's tutorial reports more than 10% higher throughput on some hardware; on the H100 with the
notebook's 64 × 64 × 32 tiles, turning grouping off made the 4,096-cubed matmul 2.5% slower (*Grouped ordering for
L2 reuse*, Figure 6.9).

## Question 8

Distinguish what `num_warps` controls from what `num_stages` controls, and relate each to a concept from Chapter 5.

**Answer.** `num_warps` sets how many warps execute one program instance: more warps give more parallelism and
latency hiding and cost registers, which can lower occupancy; it maps to Chapter 5's eligible warps and occupancy.
`num_stages` sets the software-pipelining depth of loops such as the matmul K-loop, issuing loads for later
iterations ahead of the compute; it maps to latency hiding by overlap, with a shared-memory buffer per stage.

**Explanation.** Past some depth, the shared memory per stage exhausts the SM's budget and lowers occupancy, which is
why both knobs are searched (*What num_warps and num_stages control*, Table 6.3).

## Question 9

A colleague proposes rewriting the model's main GEMM in Triton "to beat cuBLAS." Give two reasons this is likely a
poor use of effort and one situation in which a custom matmul-shaped kernel is justified.

**Answer.** First, vendor libraries are tuned per architecture and are hard to match: the chapter's fixed-tile
Triton kernel reached 36% to 54% of cuBLAS on the H100 from 1,024- to 4,096-cubed, and the autotuned kernel did not
beat it on any measured shape. Second, a kernel costs engineering, testing and maintenance for as long as it lives
in the codebase and must be re-validated with each library upgrade. A matmul-shaped kernel is justified when it
fuses work the library cannot express, such as a quantize-dequantize step fused into the matmul or fused attention.

**Explanation.** Principle #16 and *When not to write a kernel*; the cases that justify a kernel are in *When to
write a kernel* and Table 6.4.

## Question 10

A memory-bound elementwise operation owns 3% of request latency. Should you write a custom kernel for it? Justify
with both the Roofline and Amdahl's Law.

**Answer.** No, unless it can be fused with neighbors on the critical path. The Roofline says that fusion can help a
memory-bound operation, but Amdahl's Law bounds the gain: even at zero time the request becomes at most
1 / 0.97 ≈ 1.03× faster, which does not repay the cost of a kernel. Try `torch.compile` first.

**Explanation.** Principle #17 asks that the operation be on the critical path, memory-bound and fusable before a
custom kernel is written; an operation off the critical path is left alone (*When not to write a kernel*,
Table 6.4).

## Question 11

Describe how `TRITON_INTERPRET=1` changes kernel execution and why that helps debug a suspected masking error.

**Answer.** Triton runs the kernel on the CPU in Python, one program instance at a time and in order, so `print`
statements and a debugger work. You can print the offsets, the mask and the loaded tile of the last instance and
see a lane that the mask wrongly enables, for example `offsets <= n_elements`.

**Explanation.** On the GPU the off-by-one read and write pass a spot check; the interpreter shows the wrong mask
value and the out-of-bounds value it loaded (*The interpreter*).

## Question 12

Why should a kernel be validated on non-power-of-two and prime-sized inputs as well as on power-of-two shapes? What
class of bug does this catch?

**Answer.** Power-of-two shapes that are multiples of the block size leave no partial tile, so the masks are
not exercised. Odd and prime sizes create partial tiles at the boundaries and catch boundary and masking bugs,
including wrong `other` values.

**Explanation.** A kernel that passes on 1024 × 1024 and fails on 1000 × 999 has a boundary bug (*Validate against a
reference*, Principle #18).
