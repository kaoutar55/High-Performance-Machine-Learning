# Chapter 5 Knowledge check: solutions

GPU Architecture and Hardware-Aware Decisions

Section names refer to the chapter. T4 values are those of Table 5.2 and the chapter's T4 specifications.

## Question 1

A T4 decode projection runs at batch one but launches 640 thread blocks. Why is "batch one cannot fill 40 SMs" an
invalid diagnosis, and which measurements would distinguish grid under-subscription from bandwidth limitation?

**Answer.** 640 blocks are 16 per SM on the 40-SM T4, so the grid covers every SM several times over; the limit is
the low weight reuse of a matrix-vector product, about 1 FLOP per byte in FP16, which makes the kernel
bandwidth-bound. To tell the two apart, read Nsight Compute's Launch Statistics (grid size, waves per SM) and
Occupancy for under-subscription, and the achieved DRAM throughput and memory Speed of Light for bandwidth
limitation, compared with the sustained copy rate of the same GPU (234 GB/s on the case study's T4s).

**Explanation.** A matrix-vector kernel may partition its output and reduction dimensions into hundreds of blocks,
covering the device, and the actual problem is often low arithmetic intensity (*Grid size and waves*,
*Dense linear layers: decode and prefill*). Principle #11 checks grid size and waves first, and Principle #12
compares achieved bandwidth with the sustained copy rate.

## Question 2

A kernel reports 75% achieved occupancy but only 0.2 eligible warps per scheduler and low issue efficiency. Explain
why occupancy did not hide the latency and name two possible causes.

**Answer.** Occupancy measures how many warps are resident, and latency is hidden only when some of them are ready
to issue; with 0.2 eligible warps per scheduler, most resident warps are stalled. Possible causes: dependency
chains with little instruction-level parallelism, long memory latency without enough independent loads in flight,
barriers, or divergence.

**Explanation.** A kernel can have high occupancy and low throughput when every resident warp waits on the same
dependency or memory bottleneck (*Occupancy*); Table 5.3 lists "resident warps but few eligible warps" with these
interpretations and points to scheduler and warp-state statistics as the evidence.

## Question 3

An aligned warp reads 32 adjacent FP32 values. How many useful bytes are requested, and how many 32-byte sectors
normally service the request on compute capability 6.0 or later? Contrast this with a stride-8 pattern.

**Answer.** 32 × 4 = 128 useful bytes, serviced by four 32-byte sectors, so all moved bytes are used. With stride 8,
every lane touches its own sector: 32 sectors, 1,024 bytes moved for the same 128 useful bytes, 12.5% useful.

**Explanation.** An aligned warp touches min(32, 4s) sectors for stride s, so the useful fraction is max(1/s, 1/8)
(*Requested bytes vs. transferred bytes*, Figure 5.12).

## Question 4

For C = A + B in FP32, compute the useful arithmetic intensity under the one-read/one-write model. How does FP16
change the value, and why does the kernel remain low intensity?

**Answer.** One addition per element against two 4-byte reads and one 4-byte write: 1 / 12 = 0.083 FLOP/byte. In
FP16, 1 / 6 = 0.167 FLOP/byte. Both are far below the T4's ridge points (about 25 FLOP/byte for FP32 CUDA cores and
about 203 for FP16 Tensor Cores), because each element is used once.

**Explanation.** Halving precision doubles the theoretical intensity because fewer bytes move per operation, and the
operation still does one addition per element (*Low-intensity example: elementwise add*, Table 5.2).

## Question 5

Derive the weight-dominated arithmetic intensity of Y = XW when M is small. What does it predict for FP16 with
M = 1, M = 32 and a long prefill?

**Answer.** FLOPs = 2MKN and bytes ≈ s(MK + KN + MN); when M is small the KN weight term dominates, so
AI ≈ 2MKN / (sKN) = 2M / s. In FP16 (s = 2), M = 1 gives about 1 FLOP/byte and M = 32 about 32 FLOP/byte. A
512-token prefill gives far higher reuse and moves toward the compute roof, until input and output traffic,
attention and implementation details limit the simple approximation.

**Explanation.** One weight read serves M token rows, which is why continuous batching raises arithmetic intensity
(*Dense linear layers: decode and prefill*, Figure 5.16).

## Question 6

A kernel has arithmetic intensity below the FP16 Tensor Core ridge but reaches only 15% of the device-memory roof.
Give four explanations other than "the GPU needs faster arithmetic."

**Answer.** Poor coalescing, too few blocks (launch under-subscription), long dependency chains or barriers, and
insufficient memory-level parallelism. Other entries of Table 5.3 also apply: on-chip limits such as shared-memory
bank conflicts, and host gaps from launch overhead or synchronization.

**Explanation.** A kernel below the ridge has a lower memory ceiling than compute ceiling, but that does not prove
it saturates bandwidth; these causes place it far below the memory roof (*Interpreting kernel position on the
Roofline*, Table 5.3).

## Question 7

Why can a stage-level Roofline hide optimization opportunities? Describe how you would choose the dominant kernels
and aggregate their effect back to request latency.

**Answer.** A stage contains GEMMs, attention, normalizations, elementwise operations and runtime work, each with
its own FLOPs, bytes, precision and roof; one stage-level intensity averages them and hides the kernels to change.
Use the stage profile to find the kernels that own the time, place each of them with Nsight Compute, and aggregate
the effect of a change back to end-to-end latency with Amdahl's Law.

**Explanation.** This is the procedure of *Stage-level versus kernel-level intensity*, which moves from Chapter 2's
stage-level Roofline to kernel-level classification.

## Question 8

A prefill GEMM uses FP16 tensors. What additional evidence is required before claiming that Tensor Cores
accelerated it?

**Answer.** Hardware counters that report Tensor Core (HMMA) instruction activity and pipeline utilization for that
kernel, its shapes and alignment (dimensions that are multiples of 8 for FP16 on Turing), and a measured speedup
against the same operation without Tensor Cores, judged against the Tensor Core roof.

**Explanation.** Eligibility depends on the operation, the shape, layout and precision, and whether arithmetic is the
limiter; kernel names provide a hint, and counters are stronger evidence (*Tensor Core operation*, Principle #14).

## Question 9

Explain the difference between theoretical occupancy, achieved occupancy, grid waves and SM throughput. Give an
example where each pair can disagree.

**Answer.** Theoretical occupancy is the maximum fraction of warp slots a kernel could occupy, given block size,
registers, shared memory and architectural limits. Achieved occupancy is the average active-warp fraction during
execution. Waves are the total blocks divided by the blocks that can run concurrently across the device. SM
throughput is how close the SMs come to a compute pipeline ceiling. Examples: 100% theoretical occupancy with low
achieved occupancy when a grid has too few blocks or a long tail; high achieved occupancy with low SM throughput
when all resident warps are stalled on memory; one full wave that covers all SMs and still leaves SMs idle at the
end because of a long-tail block.

**Explanation.** Both occupancies describe residency and neither reports issued instructions or useful work; grid
depth and SM activity metrics answer different questions (*Occupancy*, *Grid size and waves*, *SM activity
metrics*, Table 5.1).

## Question 10

Why is BF16 generally more resistant to training overflow than FP16, and why is "BF16 cannot overflow" still
incorrect?

**Answer.** BF16 has 8 exponent bits, about the range of FP32, while FP16 has 5 exponent bits and a largest finite
value of 65,504. BF16 can still overflow at extreme magnitudes.

**Explanation.** The narrow exponent range of FP16 can cause overflow or small-gradient underflow, which is why FP16
training uses `GradScaler`; BF16 training commonly runs without loss scaling, and convergence still has to be
validated (*Floating-point formats*, *Mixed-precision training*).

## Question 11

The model is already stored in FP16. Which part of "switching to FP16 halves decode traffic" is no longer a valid new
optimization, and which mixed-precision benefit may still remain for eligible compute-bound kernels?

**Answer.** The halving of weight bytes is already in the baseline, so it is not a new saving. Faster Tensor Core
arithmetic can still help eligible compute-bound kernels, such as the prefill GEMMs with long prompts.

**Explanation.** The case study loads both models in FP16 from Chapter 2 onward and does not count FP16 storage as a
new 2× byte reduction (*Baseline dtype*); the expected effect by stage is in *Mixed precision applied to RAG stages*.

## Question 12

A decode-attention kernel slows almost linearly with context length while projection latency remains flat. Which
traffic term is likely growing, and which Chapters 8 to 9 techniques could address it?

**Answer.** KV-cache traffic: each decode step reads the stored keys and values of every earlier position, so the
bytes grow with context. Techniques: KV-cache quantization (fewer bytes per element), paged allocation (less
fragmentation and over-reservation, so more sequences fit), and the attention kernels and cache management of
Chapter 8.

**Explanation.** The dense-linear model captures weight reuse, and attention adds a separate traffic term that
becomes increasingly important at long context (*KV-cache traffic*); attention does not share the batching benefit,
because each sequence reads its own cache (*Batching and arithmetic intensity*).
