"""Google AI Studio provider implementation."""

import threading
from typing import Dict, Optional, Tuple
from .base import LLMProvider

# Real token-usage accumulator for cost measurement (e.g. the multi-turn
# checklist-standardization pilot). Keyed by (model_name, kind), kind one of
# "prompt_tokens" | "candidates_tokens" | "thoughts_tokens". Thread-safe since
# GoogleAIProvider.generate() is called concurrently from thread pools in
# several pipeline stages.
_usage_lock = threading.Lock()
_usage_totals: Dict[Tuple[str, str], int] = {}


def reset_usage_totals() -> None:
    with _usage_lock:
        _usage_totals.clear()


def get_usage_totals() -> Dict[Tuple[str, str], int]:
    with _usage_lock:
        return dict(_usage_totals)


class GoogleAIProvider(LLMProvider):
    """Google AI Studio provider using the Google Generative AI SDK.

    Uses the new google-genai package (https://ai.google.dev/gemini-api/docs/quickstart).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3-flash-preview",
        **default_params,
    ):
        """Initialize Google AI provider.

        Args:
            api_key: Google AI Studio API key. If not provided, uses GEMINI_API_KEY env var.
            model_name: Model name (e.g., 'gemini-3-flash-preview', 'gemini-3-pro-preview')
            **default_params: Default generation parameters
        """
        self.api_key = api_key
        self.model_name = model_name
        self.default_params = default_params
        self._client = None
        # Guards lazy init so concurrent first-calls (pipeline thread pool) don't
        # each build a Client; an orphaned Client gets GC'd and closes its shared
        # httpx pool, surfacing as "client has been closed" in the live thread.
        self._client_lock = threading.Lock()

    def _initialize_client(self):
        """Lazy, thread-safe initialization of the Google AI client."""
        if self._client is not None:
            return
        with self._client_lock:
            if self._client is not None:
                return
            try:
                from google import genai

                # Client gets API key from GEMINI_API_KEY env var if not provided
                client_kwargs = {}
                if self.api_key:
                    client_kwargs["api_key"] = self.api_key

                # Check for http_options in default_params
                if "http_options" in self.default_params:
                    client_kwargs["http_options"] = self.default_params["http_options"]

                self._client = genai.Client(**client_kwargs)
            except ImportError:
                raise ImportError(
                    "Google Generative AI SDK not installed. "
                    "Install it with: pip install google-genai"
                )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Generate text using Google AI Studio.

        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt (mapped to system_instruction)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters (e.g., thinking_config)

        Returns:
            Generated text
        """
        self._initialize_client()

        # Build generation config
        from google.genai import types

        config_params = {**self.default_params}
        # http_options is a client-level setting (e.g. request timeout); it must not
        # be forwarded into GenerateContentConfig.
        config_params.pop("http_options", None)

        # Handle parameter mapping (max_tokens -> max_output_tokens)
        if "max_tokens" in config_params:
            config_params["max_output_tokens"] = config_params.pop("max_tokens")

        if temperature is not None:
            config_params["temperature"] = temperature
        if max_tokens is not None:
            config_params["max_output_tokens"] = max_tokens

        # Extract special parameters
        # System instruction
        system_instruction = kwargs.pop("system_instruction", system_prompt)

        # Thinking config: Check both kwargs and config_params (defaults)
        # We pop from config_params to avoid "multiple values" error
        default_thinking = config_params.pop("thinking_config", None)
        thinking_config = kwargs.pop("thinking_config", default_thinking)

        # If thinking_config is a dict, wrap it in types.ThinkingConfig
        if isinstance(thinking_config, dict):
            # Map 'thinking_level' or 'include_thoughts' from dict
            t_level = thinking_config.get("thinking_level")
            inc_thoughts = thinking_config.get("include_thoughts")

            # Construct dictionary for ThinkingConfig constructor, filtering None
            tc_kwargs = {}
            if t_level is not None:
                tc_kwargs["thinking_level"] = t_level
            if inc_thoughts is not None:
                tc_kwargs["include_thoughts"] = inc_thoughts

            if tc_kwargs:
                thinking_config = types.ThinkingConfig(**tc_kwargs)

        # Add any remaining kwargs
        config_params.update(kwargs)

        # Create config
        if system_instruction or thinking_config:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                thinking_config=thinking_config,
                **config_params,
            )
        else:
            config = types.GenerateContentConfig(**config_params)

        try:
            response = self._client.models.generate_content(
                model=self.model_name, contents=prompt, config=config
            )
        except Exception as e:
            raise RuntimeError(f"Google AI generation failed: {str(e)}")

        try:
            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                with _usage_lock:
                    key = self.model_name
                    _usage_totals[(key, "prompt_tokens")] = _usage_totals.get(
                        (key, "prompt_tokens"), 0
                    ) + (getattr(usage, "prompt_token_count", None) or 0)
                    _usage_totals[(key, "candidates_tokens")] = _usage_totals.get(
                        (key, "candidates_tokens"), 0
                    ) + (getattr(usage, "candidates_token_count", None) or 0)
                    _usage_totals[(key, "thoughts_tokens")] = _usage_totals.get(
                        (key, "thoughts_tokens"), 0
                    ) + (getattr(usage, "thoughts_token_count", None) or 0)
        except Exception:  # noqa: BLE001 — instrumentation must never break generate()'s str contract
            pass

        return response.text

    def get_provider_name(self) -> str:
        """Get provider name."""
        return f"Google AI Studio ({self.model_name})"
