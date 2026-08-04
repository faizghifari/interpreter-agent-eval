"""OpenAI provider implementation."""

import re
from typing import Optional
from .base import LLMProvider

# Chat-template control tokens that some OpenAI-compatible local backends (e.g. LM
# Studio) fail to register as a model's actual stop token, so they leak into
# `message.content` verbatim instead of being stripped server-side. Observed in
# practice: 100% of Arabic user-simulation responses (c4ai-command-r7b-arabic-02-2025
# via LM Studio) ended with a literal "<|END_RESPONSE|>" string.
_LEAKED_STOP_TOKEN_RE = re.compile(
    r"<\|.*?\|>|\[END[_ ]?RESPONSE\]|<end_of_turn>|<\|im_end\|>|\[/?INST\]|<<SYS>>"
)


def _strip_leaked_stop_tokens(text: str) -> str:
    return _LEAKED_STOP_TOKEN_RE.sub("", text).strip()


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (also compatible with OpenAI-compatible endpoints like LM Studio, Ollama)."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        base_url: Optional[str] = None,
        **default_params,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (use any non-empty string for local servers)
            model_name: Model name (e.g., 'gpt-3.5-turbo', 'gpt-4')
            base_url: Optional base URL for OpenAI-compatible endpoints (e.g., LM Studio, Ollama)
            **default_params: Default generation parameters
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.default_params = default_params
        self._client = None

    def _initialize_client(self):
        """Lazy initialization of the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                raise ImportError(
                    "OpenAI SDK not installed. " "Install it with: pip install openai"
                )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Generate text using OpenAI API.

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
            return _strip_leaked_stop_tokens(response.choices[0].message.content)
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {str(e)}")

    def get_provider_name(self) -> str:
        """Get provider name."""
        return f"OpenAI ({self.model_name})"
