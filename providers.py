"""Model providers — pluggable backends for the dojo agent.

Supports:
1. API models (OpenAI, Azure OpenAI, GitHub Models)
2. Self-hosted models via vLLM/SGLang (running on AKS GPU nodes)
3. Local HuggingFace models (for small models)

The dojo agent uses whichever provider you configure.
For RL training, you'd use a self-hosted model so you can update weights.
"""

import os
from typing import Optional


class ModelProvider:
    """Base class for model providers."""

    def generate(self, messages: list[dict], temperature: float = 0.2) -> str:
        """Generate a completion from chat messages. Returns the response text."""
        raise NotImplementedError


class OpenAIProvider(ModelProvider):
    """OpenAI-compatible API (works with Azure OpenAI, GitHub Models, vLLM)."""

    def __init__(self, model: str = "gpt-4o",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    def generate(self, messages: list[dict], temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature,
        )
        return response.choices[0].message.content.strip()


class VLLMProvider(ModelProvider):
    """Self-hosted model via vLLM on AKS GPU nodes.

    vLLM exposes an OpenAI-compatible API, so this is a thin wrapper
    that knows how to discover the vLLM endpoint in the cluster.

    Usage:
        # If vLLM is running as a k8s service:
        provider = VLLMProvider(
            model="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            base_url="http://vllm-svc.default.svc.cluster.local:8000/v1"
        )

        # Or port-forwarded locally:
        provider = VLLMProvider(
            model="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            base_url="http://localhost:8000/v1"
        )
    """

    def __init__(self, model: str, base_url: str):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(api_key="not-needed", base_url=base_url)

    def generate(self, messages: list[dict], temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature,
        )
        return response.choices[0].message.content.strip()


class HuggingFaceProvider(ModelProvider):
    """Local HuggingFace model (for small models that fit in memory).

    Good for experimentation with models you're fine-tuning locally.
    For production RL training, use VLLMProvider instead.
    """

    def __init__(self, model_id: str = "deepseek-ai/deepseek-coder-1.3b-instruct",
                 device: str = "auto"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, messages: list[dict], temperature: float = 0.2) -> str:
        import torch

        # Format messages into a prompt
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n\n"
        prompt += "Assistant: "

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=max(temperature, 0.01),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
        return generated.strip()


def create_provider(provider_type: str = "auto", **kwargs) -> ModelProvider:
    """Factory function to create a model provider.

    Args:
        provider_type: "openai", "vllm", "huggingface", or "auto"
        **kwargs: Provider-specific arguments

    Auto-detection:
        - If VLLM_BASE_URL is set → VLLMProvider
        - If OPENAI_API_KEY is set → OpenAIProvider
        - Otherwise → error
    """
    if provider_type == "auto":
        vllm_url = os.environ.get("VLLM_BASE_URL")
        if vllm_url:
            model = kwargs.get("model", os.environ.get("VLLM_MODEL", "default"))
            return VLLMProvider(model=model, base_url=vllm_url)

        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            return OpenAIProvider(
                model=kwargs.get("model", os.environ.get("OPENAI_MODEL", "gpt-4o")),
                api_key=api_key,
                base_url=os.environ.get("OPENAI_BASE_URL"),
            )

        raise ValueError(
            "Set VLLM_BASE_URL (for self-hosted) or OPENAI_API_KEY (for API models)"
        )

    elif provider_type == "openai":
        return OpenAIProvider(**kwargs)
    elif provider_type == "vllm":
        return VLLMProvider(**kwargs)
    elif provider_type == "huggingface":
        return HuggingFaceProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider_type}")
