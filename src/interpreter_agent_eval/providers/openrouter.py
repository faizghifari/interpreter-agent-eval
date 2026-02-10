"""OpenRouter provider implementation."""

from typing import Optional
from .base import LLMProvider


class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider for accessing various models."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str,
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
        **default_params
    ):
        """Initialize OpenRouter provider.
        
        Args:
            api_key: OpenRouter API key
            model_name: Model identifier (e.g., 'anthropic/claude-2', 'meta-llama/llama-2-70b-chat')
            site_url: Optional site URL for attribution
            app_name: Optional app name for attribution
            **default_params: Default generation parameters
        """
        self.api_key = api_key
        self.model_name = model_name
        self.site_url = site_url
        self.app_name = app_name
        self.default_params = default_params
        self._client = None
    
    def _initialize_client(self):
        """Lazy initialization of the OpenRouter client."""
        if self._client is None:
            try:
                from openai import OpenAI
                # OpenRouter uses OpenAI-compatible API
                self._client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.api_key
                )
            except ImportError:
                raise ImportError(
                    "OpenAI SDK not installed (required for OpenRouter). "
                    "Install it with: pip install openai"
                )
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text using OpenRouter API.
        
        Args:
            prompt: Input prompt
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
            params['max_tokens'] = max_tokens
        if temperature is not None:
            params['temperature'] = temperature
        params.update(kwargs)
        
        # Add OpenRouter-specific headers
        extra_headers = {}
        if self.site_url:
            extra_headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            extra_headers["X-Title"] = self.app_name
        
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                extra_headers=extra_headers if extra_headers else None,
                **params
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenRouter generation failed: {str(e)}")
    
    def get_provider_name(self) -> str:
        """Get provider name."""
        return f"OpenRouter ({self.model_name})"
