# LLMCoreLearning
Learning LLM components ground up


Components/features of LLM
Coverage by layer of the stack:
Stage, 	Entries 
Tokenization & Input,	BPE, SentencePiece/Unigram, Embeddings (1–3)
Positional Encoding	RoPE, ALiBi (4–5)
Attention	MHA, GQA/MQA, FlashAttention, Sliding Window/Sparse (6–9)
Architecture Blocks	MoE Routing, SwiGLU, RMSNorm, MTP, Mamba/Hybrids (10–11, 12, 38–39)
Pretraining	AdamW, LR Schedules, Mixed Precision, Grad Checkpointing, ZeRO/FSDP, 3D Parallelism, Chinchilla Scaling Laws, Data Curation (13–20)
Inference & Serving	KV Cache, Continuous Batching/PagedAttention, Quantization (GPTQ/AWQ/FP8), Speculative Decoding, CUDA Graphs/Compilation, YaRN Long-Context (21–26)
Alignment	SFT, RLHF/PPO, DPO/KTO, Constitutional AI/RLAIF, Reward Models, BoN Distillation, RLVR/GRPO (27–31, 40, 45)
Capabilities & Features	Function Calling/JSON Mode, CoT/Few-Shot, RAG, Safety Classifiers, Evaluation Suites, Interpretability (SAEs, Probes), LoRA/QLoRA, Model Souping, Distillation (32–37, 41–43)
System Integration	Entry #44 — the full end-to-end reference architecture tying every component into one production stack (training → alignment → serving → safety → eval), including cost/latency trade-offs


Todo
1. Include comparative examples of FAISS vs TurboVector/TurboQuant (Google's paper April 2025)
2. GeoAI demos to be validated and enhanced. i think it better to publish in a separate repo due to size
