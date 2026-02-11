"""vLLM provider for HuggingFace models."""

from typing import Optional
from .base import LLMProvider


class VLLMProvider(LLMProvider):
    """vLLM provider for serving HuggingFace models."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        **default_params,
    ):
        """Initialize vLLM provider.

        Args:
            base_url: Base URL of the vLLM server (e.g., 'http://localhost:8000')
            model_name: Model name served by vLLM
            api_key: Optional API key if authentication is required
            **default_params: Default generation parameters
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.default_params = default_params
        self._client = None

    def _initialize_client(self):
        """Lazy initialization of the vLLM client."""
        if self._client is None:
            try:
                from openai import OpenAI

                # vLLM provides OpenAI-compatible API
                self._client = OpenAI(
                    base_url=f"{self.base_url}/v1", api_key=self.api_key or "EMPTY"
                )
            except ImportError:
                raise ImportError(
                    "OpenAI SDK not installed (required for vLLM client). "
                    "Install it with: pip install openai"
                )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Generate text using vLLM server.

        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        self._initialize_client()

        # Merge parameters
        params = {**self.default_params}
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature
        params.update(kwargs)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self.model_name, messages=messages, **params
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"vLLM generation failed: {str(e)}")

    def get_provider_name(self) -> str:
        """Get provider name."""
        return f"vLLM ({self.model_name})"
