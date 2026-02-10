"""Google AI Studio provider implementation."""

from typing import Optional, Dict, Any
from .base import LLMProvider


class GoogleAIProvider(LLMProvider):
    """Google AI Studio provider using the Google Generative AI SDK."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-pro",
        **default_params
    ):
        """Initialize Google AI provider.
        
        Args:
            api_key: Google AI Studio API key
            model_name: Model name (e.g., 'gemini-pro', 'gemini-pro-vision')
            **default_params: Default generation parameters
        """
        self.api_key = api_key
        self.model_name = model_name
        self.default_params = default_params
        self._client = None
    
    def _initialize_client(self):
        """Lazy initialization of the Google AI client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model_name)
            except ImportError:
                raise ImportError(
                    "Google Generative AI SDK not installed. "
                    "Install it with: pip install google-generativeai"
                )
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text using Google AI Studio.
        
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
        generation_config = {**self.default_params}
        if max_tokens is not None:
            generation_config['max_output_tokens'] = max_tokens
        if temperature is not None:
            generation_config['temperature'] = temperature
        generation_config.update(kwargs)
        
        try:
            response = self._client.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"Google AI generation failed: {str(e)}")
    
    def get_provider_name(self) -> str:
        """Get provider name."""
        return f"Google AI Studio ({self.model_name})"
