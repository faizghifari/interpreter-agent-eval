"""FriendliAI provider implementation."""

import os
from typing import Optional
from .base import LLMProvider


class FriendliProvider(LLMProvider):
    """FriendliAI provider using OpenAI-compatible API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "LGAI-EXAONE/K-EXAONE-236B-A23B",
        base_url: str = "https://api.friendli.ai/serverless/v1",
        timeout: float = 1000.0,
        **default_params,
    ):
        """Initialize Friendli provider.

        Args:
            api_key: Friendli API Token. If not provided, uses FRIENDLI_TOKEN env var.
            model_name: Model name.
            base_url: Base URL for Friendli API.
            timeout: Timeout for requests in seconds. Default 1000s.
            **default_params: Default generation parameters
        """
        self.api_key = api_key or os.getenv("FRIENDLI_TOKEN")
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        self.default_params = default_params
        self._client = None

    def _initialize_client(self):
        """Lazy initialization of the OpenAI client for Friendli."""
        if self._client is None:
            try:
                from openai import OpenAI

                if not self.api_key:
                    raise ValueError(
                        "Friendli API key is required. Set FRIENDLI_TOKEN env var or pass api_key."
                    )

                self._client = OpenAI(
                    api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
                )
            except ImportError:
                raise ImportError(
                    "OpenAI SDK not installed. Install it with: pip install openai"
                )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Generate text using Friendli API.

        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters (e.g. enable_thinking)

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

        # Handle thinking parameter: Check both kwargs and params (defaults)
        enable_thinking = kwargs.pop(
            "enable_thinking", params.pop("enable_thinking", None)
        )
        # Also check alias "thinking"
        if enable_thinking is None:
            enable_thinking = kwargs.pop("thinking", params.pop("thinking", None))

        # Prepare extra_body if needed
        extra_body = kwargs.pop("extra_body", params.pop("extra_body", {}))

        if enable_thinking:
            # Check if extra_body already has it, otherwise add it
            if "chat_template_kwargs" not in extra_body:
                extra_body["chat_template_kwargs"] = {}
            extra_body["chat_template_kwargs"]["enable_thinking"] = True
            # Also parse_reasoning usually needed to get the content clean?
            # User example: "parse_reasoning": True
            extra_body["parse_reasoning"] = True

        if extra_body:
            params["extra_body"] = extra_body

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
            raise RuntimeError(f"Friendli generation failed: {str(e)}")

    def get_provider_name(self) -> str:
        """Get provider name."""
        return f"Friendli ({self.model_name})"
