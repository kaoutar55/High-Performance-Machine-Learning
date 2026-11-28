# Chapter 10 online resources

Efficient Model Adaptation and Maintenance

This document collects the online resources that the print chapter refers to: the answers to the knowledge check and the references cited in the chapter, numbered as in the text. The tables and notes moved out of the print chapter are in the [chapter README](README.md#online-resources).

## Knowledge check answers

The same answers are in [`knowledge_check_solutions.md`](knowledge_check_solutions.md).

Section names refer to the chapter.

### Question 1

Using 16 bytes per parameter for full fine-tuning, 2 bytes per frozen parameter and 16 bytes per adapter parameter
for LoRA, estimate the static training memory of a 3B-parameter model with a rank-16 adapter of 15 million
parameters. Which method fits on a 24 GB GPU?

**Answer.** Full fine-tuning: 3 × 10⁹ × 16 bytes = 48 GB. LoRA: 3 × 10⁹ × 2 bytes + 15 × 10⁶ × 16 bytes =
6 GB + 0.24 GB, about 6.2 GB. LoRA fits on a 24 GB GPU with room for activations; full fine-tuning does not fit.

**Explanation.** The counts are those of Code 10.1 and Principle #38; activations come on top of each figure
(*Memory of full fine-tuning*, *Low-rank adaptation*).

### Question 2

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

### Question 3

A merged adapter and an unmerged adapter produce the same outputs. When would you serve the adapter unmerged, and
what would you measure before deciding?

**Answer.** Serve unmerged when many adapters share one base model, for example one per customer, language or task,
so that one copy of the base weights serves all of them. Measure TPOT, capacity and TTFT at capacity with the engine
version you deploy, with the number of distinct adapters per batch that your traffic produces.

**Explanation.** In Table 10.13 (vLLM 0.33.0, H100) the merged adapter matched the base model (6.45 against 6.47 ms
TPOT, 64 requests per second), one unmerged adapter raised TPOT by 22% and lowered capacity to 48, and 16 distinct
adapters per batch lowered capacity to 24 (*Serving fine-tuned adapters*, *Results: serving the adapters*,
Principle #39).

### Question 4

Why can token-level distillation transfer knowledge from Llama-3.1-8B to Llama-3.2-1B but not to Gemma-2B? Which
form of distillation works for the second pair?

**Answer.** Token-level distillation matches the student's distribution to the teacher's logits over the student's
vocabulary, which needs a shared tokenizer; Llama-3.1-8B and Llama-3.2-1B share a 128,256-token vocabulary, and Llama
and Gemma do not. For Gemma-2B only the teacher's generated text can be transferred: sequence-level, black-box
distillation, in which the student is fine-tuned on the teacher's outputs.

**Explanation.** *Knowledge distillation* and *Distillation methods for LLMs*; the case study trains Gemma-2B on the
teacher's text only.

### Question 5

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

## References

The 65 references cited in Chapter 10, numbered as in the text.

**[1]** vLLM project, "v0.33.0" release notes, 2026. <https://github.com/vllm-project/vllm/releases>

**[2]** Google, "Gemma 4: Expanding the Gemmaverse with Apache 2.0," Google Open Source Blog, 2026. <https://opensource.googleblog.com/2026/03/gemma-4-expanding-the-gemmaverse-with-apache-20.html>

**[3]** OpenAI, "Deprecations," API documentation (fine-tuned models and their base models). <https://developers.openai.com/api/docs/deprecations>

**[4]** Chen, L., Zaharia, M., and Zou, J., "How Is ChatGPT's Behavior Changing Over Time?," Harvard Data Science Review, 2024. <https://arxiv.org/abs/2307.09009>

**[5]** Meta, "Llama 3.1 model card" (knowledge cutoff, training tokens and GPU hours). <https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md>

**[6]** Cheng, J., Marone, M., Weller, O., Lawrie, D., Khashabi, D., and Van Durme, B., "Dated Data: Tracing Knowledge Cutoffs in Large Language Models," COLM 2024. <https://arxiv.org/abs/2403.12958>

**[7]** Luo, Y., Yang, Z., Meng, F., Li, Y., Zhou, J., and Zhang, Y., "An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning," 2023. <https://arxiv.org/abs/2308.08747>

**[8]** Biderman, D., et al., "LoRA Learns Less and Forgets Less," Transactions on Machine Learning Research, 2024. <https://arxiv.org/abs/2405.09673>

**[9]** Shuttleworth, R., Andreas, J., Torralba, A., and Sharma, P., "LoRA vs Full Fine-tuning: An Illusion of Equivalence," NeurIPS 2025. <https://arxiv.org/abs/2410.21228>

**[10]** Gekhman, Z., et al., "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?," EMNLP 2024. <https://aclanthology.org/2024.emnlp-main.444>

**[11]** Sculley, D., et al., "Hidden Technical Debt in Machine Learning Systems," NeurIPS 2015. <https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems>

**[12]** Dmonte, A., et al., "Improving Training Efficiency and Reducing Maintenance Costs via Language Specific Model Merging," 2026. <https://arxiv.org/abs/2601.16127>

**[13]** Hu, E. J., et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022. <https://arxiv.org/abs/2106.09685>

**[14]** Schulman, J., and Thinking Machines Lab, "LoRA Without Regret," Connectionism, 2025. <https://thinkingmachines.ai/blog/lora/>

**[15]** Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L., "QLoRA: Efficient Finetuning of Quantized LLMs," NeurIPS 2023. <https://arxiv.org/abs/2305.14314>

**[16]** Kalajdzievski, D., "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA," 2023. <https://arxiv.org/abs/2312.03732>

**[17]** bitsandbytes contributors, "bitsandbytes" release notes (v0.48–v0.50). <https://github.com/bitsandbytes-foundation/bitsandbytes/releases>

**[18]** Hugging Face, "LoRA," PEFT developer guide (v0.21). <https://huggingface.co/docs/peft/developer_guides/lora>

**[19]** Hugging Face, "TRL: Transformer Reinforcement Learning," documentation, GKD Trainer (v1.17). <https://huggingface.co/docs/trl/main/en/gkd_trainer>

**[20]** Meta PyTorch, "torchtune" README (project status). <https://github.com/meta-pytorch/torchtune>

**[21]** Liu, S.-Y., et al., "DoRA: Weight-Decomposed Low-Rank Adaptation," ICML 2024. <https://arxiv.org/abs/2402.09353>

**[22]** Hayou, S., Ghosh, N., and Yu, B., "LoRA+: Efficient Low Rank Adaptation of Large Models," ICML 2024. <https://arxiv.org/abs/2402.12354>

**[23]** Meng, F., Wang, Z., and Zhang, M., "PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models," NeurIPS 2024. <https://arxiv.org/abs/2404.02948>

**[24]** Kopiczko, D. J., Blankevoort, T., and Asano, Y. M., "VeRA: Vector-based Random Matrix Adaptation," ICLR 2024. <https://arxiv.org/abs/2310.11454>

**[25]** Zhang, L., Zhang, L., Shi, S., Chu, X., and Li, B., "LoRA-FA: Memory-efficient Low-rank Adaptation for Large Language Models Fine-tuning," 2023. <https://arxiv.org/abs/2308.03303>

**[26]** Zhao, J., et al., "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection," ICML 2024. <https://arxiv.org/abs/2403.03507>

**[27]** Zhao, J., et al. (Predibase), "LoRA Land: 310 Fine-tuned LLMs that Rival GPT-4, A Technical Report," 2024. <https://arxiv.org/abs/2405.00732>

**[28]** Chen, L., et al., "Punica: Multi-Tenant LoRA Serving," MLSys 2024. <https://arxiv.org/abs/2310.18547>

**[29]** Sheng, Y., et al., "S-LoRA: Serving Thousands of Concurrent LoRA Adapters," MLSys 2024. <https://arxiv.org/abs/2311.03285>

**[30]** Amazon Web Services, "Efficiently Serve Dozens of Fine-Tuned Models with vLLM on Amazon SageMaker AI and Amazon Bedrock," AWS Machine Learning Blog, 2026. <https://aws.amazon.com/blogs/machine-learning/efficiently-serve-dozens-of-fine-tuned-models-with-vllm-on-amazon-sagemaker-ai-and-amazon-bedrock>

**[31]** Fireworks AI, "Understanding LoRA Performance," documentation. <https://docs.fireworks.ai/guides/understanding_lora_performance>

**[32]** vLLM project, "LoRA Adapters," vLLM documentation. <https://docs.vllm.ai/en/latest/features/lora/>

**[33]** Hinton, G., Vinyals, O., and Dean, J., "Distilling the Knowledge in a Neural Network," 2015. <https://arxiv.org/abs/1503.02531>

**[34]** Gu, Y., Dong, L., Wei, F., and Huang, M., "MiniLLM: Knowledge Distillation of Large Language Models," ICLR 2024. <https://arxiv.org/abs/2306.08543>

**[35]** Kim, Y., and Rush, A. M., "Sequence-Level Knowledge Distillation," EMNLP 2016. <https://arxiv.org/abs/1606.07947>

**[36]** Agarwal, R., et al., "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes," ICLR 2024. <https://arxiv.org/abs/2306.13649>

**[37]** Ko, J., Kim, S., Chen, T., and Yun, S.-Y., "DistiLLM: Towards Streamlined Distillation for Large Language Models," ICML 2024. <https://arxiv.org/abs/2402.03898>

**[38]** DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning," 2025. <https://arxiv.org/abs/2501.12948>

**[39]** Lu, K., and Thinking Machines Lab, "On-Policy Distillation," Connectionism, 2025. <https://thinkingmachines.ai/blog/on-policy-distillation/>

**[40]** Google, "Gemma 3 QAT Models: Bringing State-of-the-Art AI to Consumer GPUs," Google Developers Blog, 2025. <https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/>

**[41]** Meta, "Introducing Quantized Llama Models with Increased Speed and a Reduced Memory Footprint," 2024. <https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/>

**[42]** Liu, Z., et al., "LLM-QAT: Data-Free Quantization Aware Training for Large Language Models," Findings of ACL 2024. <https://arxiv.org/abs/2305.17888>

**[43]** Han, S., Pool, J., Tran, J., and Dally, W. J., "Learning both Weights and Connections for Efficient Neural Networks," NeurIPS 2015. <https://arxiv.org/abs/1506.02626>

**[44]** Frankle, J., and Carbin, M., "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks," ICLR 2019. <https://arxiv.org/abs/1803.03635>

**[45]** Michel, P., Levy, O., and Neubig, G., "Are Sixteen Heads Really Better than One?," NeurIPS 2019. <https://arxiv.org/abs/1905.10650>

**[46]** Men, X., et al., "ShortGPT: Layers in Large Language Models are More Redundant Than You Expect," Findings of ACL 2025. <https://arxiv.org/abs/2403.03853>

**[47]** Gromov, A., et al., "The Unreasonable Ineffectiveness of the Deeper Layers," ICLR 2025. <https://arxiv.org/abs/2403.17887>

**[48]** Xia, M., Gao, T., Zeng, Z., and Chen, D., "Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning," ICLR 2024. <https://arxiv.org/abs/2310.06694>

**[49]** Muralidharan, S., et al., "Compact Language Models via Pruning and Knowledge Distillation," NeurIPS 2024. <https://arxiv.org/abs/2407.14679>

**[50]** NVIDIA, "NVIDIA Blackwell Platform Sets New LLM Inference Records in MLPerf Inference v4.1," NVIDIA Technical Blog, 2024. <https://developer.nvidia.com/blog/nvidia-blackwell-platform-sets-new-llm-inference-records-in-mlperf-inference-v4-1>

**[51]** Sreenivas, S. T., et al., "LLM Pruning and Distillation in Practice: The Minitron Approach," 2024. <https://arxiv.org/abs/2408.11796>

**[52]** Meta, "Llama 3.2 model card" (pruning and distillation from Llama 3.1 8B and 70B). <https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md>

**[53]** Gemma Team, Google DeepMind, "Gemma 2: Improving Open Language Models at a Practical Size," 2024. <https://arxiv.org/abs/2408.00118>

**[54]** Yuan, Z., et al., "ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models," 2023. <https://arxiv.org/abs/2312.05821>

**[55]** Wang, X., Zheng, Y., Wan, Z., and Zhang, M., "SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression," ICLR 2025. <https://arxiv.org/abs/2403.07378>

**[56]** Liu, J., et al., "BitDelta: Your Fine-Tune May Only Be Worth One Bit," NeurIPS 2024. <https://arxiv.org/abs/2402.10193>

**[57]** Busbridge, D., et al., "Distillation Scaling Laws," ICML 2025. <https://proceedings.mlr.press/v267/busbridge25a.html>

**[58]** Ovadia, O., Brief, M., Mishaeli, M., and Elisha, O., "Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs," EMNLP 2024. <https://arxiv.org/abs/2312.05934>

**[59]** Zhang, T., et al., "RAFT: Adapting Language Model to Domain Specific RAG," COLM 2024. <https://arxiv.org/abs/2403.10131>

**[60]** Wang, R., et al., "Trans-LoRA: Towards Data-free Transferable Parameter Efficient Finetuning," NeurIPS 2024. <https://arxiv.org/abs/2405.17258>

**[61]** Xia, F., Liao, M., et al., "Cross-LoRA: A Data-Free LoRA Transfer Framework across Heterogeneous LLMs," 2025. <https://arxiv.org/abs/2508.05232>

**[62]** Ibrahim, A., et al., "Simple and Scalable Strategies to Continually Pre-train Large Language Models," Transactions on Machine Learning Research, 2024. <https://arxiv.org/abs/2403.08763>

**[63]** Yang, Z., et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering," EMNLP 2018. <https://arxiv.org/abs/1809.09600>

**[64]** Rajpurkar, P., Jia, R., and Liang, P., "Know What You Don't Know: Unanswerable Questions for SQuAD," ACL 2018. <https://arxiv.org/abs/1806.03822>

**[65]** Qwen Team, "Qwen3-VL-8B-Instruct," model card. <https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct>
