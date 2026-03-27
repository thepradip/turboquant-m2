"""
Integration adapters for LLM inference frameworks.

Each integration is lazily imported to avoid pulling in heavy dependencies
when they aren't needed.

Available integrations:
  - turboquant.integrations.transformers_adapter: HuggingFace Transformers
  - turboquant.integrations.ollama_adapter: Ollama REST API
  - turboquant.integrations.vllm_adapter: vLLM
  - turboquant.integrations.llamacpp_adapter: llama.cpp / GGUF
"""
