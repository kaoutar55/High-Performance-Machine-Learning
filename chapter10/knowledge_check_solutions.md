# Chapter 10 Knowledge check: solutions

Efficient Model Adaptation and Maintenance

Section names refer to the chapter.

## Question 1

Using 16 bytes per parameter for full fine-tuning, 2 bytes per frozen parameter and 16 bytes per adapter parameter
for LoRA, estimate the static training memory of a 3B-parameter model with a rank-16 adapter of 15 million
parameters. Which method fits on a 24 GB GPU?

**Answer.** Full fine-tuning: 3 × 10⁹ × 16 bytes = 48 GB. LoRA: 3 × 10⁹ × 2 bytes + 15 × 10⁶ × 16 bytes =
6 GB + 0.24 GB, about 6.2 GB. LoRA fits on a 24 GB GPU with room for activations; full fine-tuning does not fit.

**Explanation.** The counts are those of Code 10.1 and Principle #38; activations come on top of each figure
(*Memory of full fine-tuning*, *Low-rank adaptation*).

## Question 2

Why does a LoRA training step with gradient checkpointing cost about three quarters of the FLOPs of a full
fine-tuning step? Which extra work does QLoRA add, and how would you measure its cost on your GPU?

**Answer.** Per token, the forward pass costs 2N FLOPs, checkpointing recomputes it (2N), and the backward pass costs
2N for the gradients with respect to the inputs plus, when the weights are trained, 2N for the gradients with
respect to the weights. Full fine-tuning totals 8N; LoRA skips the weight gradients of the frozen base and totals
6N, three quarters. QLoRA dequantizes each block of 4-bit base weights to BF16 in every forward and backward pass.
Time a short run of LoRA and QLoRA on your GPU with your library version and compare tokens per second.

**Explanation.** *Estimating the time of a fine-tuning run*, Code 10.3, and Principle #38. In the case study QLoRA
processed 1.8% to 4.4% more tokens per second than LoRA (bitsandbytes 0.50.2, with paged 8-bit AdamW against LoRA's
fused AdamW), so the cost of dequantization depends on the GPU and the library version.

## Question 3

A merged adapter and an unmerged adapter produce the same outputs. When would you serve the adapter unmerged, and
what would you measure before deciding?

**Answer.** Serve unmerged when many adapters share one base model, for example one per customer, language or task,
so that one copy of the base weights serves all of them. Measure TPOT, capacity and TTFT at capacity with the engine
version you deploy, with the number of distinct adapters per batch that your traffic produces.

**Explanation.** In Table 10.13 (vLLM 0.33.0, H100) the merged adapter matched the base model (6.45 against 6.47 ms
TPOT, 64 requests per second), one unmerged adapter raised TPOT by 22% and lowered capacity to 48, and 16 distinct
adapters per batch lowered capacity to 24 (*Serving fine-tuned adapters*, *Results: serving the adapters*,
Principle #39).

## Question 4

Why can token-level distillation transfer knowledge from Llama-3.1-8B to Llama-3.2-1B but not to Gemma-2B? Which
form of distillation works for the second pair?

**Answer.** Token-level distillation matches the student's distribution to the teacher's logits over the student's
vocabulary, which needs a shared tokenizer; Llama-3.1-8B and Llama-3.2-1B share a 128,256-token vocabulary, and Llama
and Gemma do not. For Gemma-2B only the teacher's generated text can be transferred: sequence-level, black-box
distillation, in which the student is fine-tuned on the teacher's outputs.

**Explanation.** *Knowledge distillation* and *Distillation methods for LLMs*; the case study trains Gemma-2B on the
teacher's text only.

## Question 5

A company's internal documents change every week, and users complain that the RAG assistant answers in the wrong
format. Which update would you apply to each problem, and what would your release gate measure?

**Answer.** Put the changing documents in the retrieval index, re-indexed each week, and fix the format with a
prompt change or, if that is not enough, a LoRA adapter trained on examples in the required format. The release gate
measures target quality as a paired difference with the current model, forgetting on a fixed suite of tasks outside
the target domain together with the KL divergence on a fixed prompt set, serving performance (TTFT, TPOT and
capacity under the load test of Chapter 8), and stores the data version, configuration and results for
reproducibility.

**Explanation.** New facts belong in the retrieval index and new behaviors in the weights (Principle #41, *New facts
and new behaviors*); in the case study an adapter trained on articles answered 2.5% of questions about them
closed-book, against 79.0% for the base model with the articles retrieved. The gate is described in *Release gates*.
