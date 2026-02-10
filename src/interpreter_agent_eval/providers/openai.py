"""OpenAI provider implementation."""

from typing import Optional
from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        **default_params
    ):
        """Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model_name: Model name (e.g., 'gpt-3.5-turbo', 'gpt-4')
            **default_params: Default generation parameters
        """
        self.api_key = api_key
        self.model_name = model_name
        self.default_params = default_params
        self._client = None
    
    def _initialize_client(self):
        """Lazy initialization of the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "OpenAI SDK not installed. "
                    "Install it with: pip install openai"
                )
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text using OpenAI API.
        
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
        
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **params
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {str(e)}")
    
    def get_provider_name(self) -> str:
        """Get provider name."""
        return f"OpenAI ({self.model_name})"
