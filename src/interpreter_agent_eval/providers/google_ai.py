"""Google AI Studio provider implementation."""

from typing import Optional
from .base import LLMProvider


class GoogleAIProvider(LLMProvider):
    """Google AI Studio provider using the Google Generative AI SDK.
    
    Uses the new google-genai package (https://ai.google.dev/gemini-api/docs/quickstart).
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp",
        **default_params
    ):
        """Initialize Google AI provider.
        
        Args:
            api_key: Google AI Studio API key. If not provided, uses GEMINI_API_KEY env var.
            model_name: Model name (e.g., 'gemini-2.0-flash-exp', 'gemini-1.5-flash')
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
                from google import genai
                # Client gets API key from GEMINI_API_KEY env var if not provided
                if self.api_key:
                    self._client = genai.Client(api_key=self.api_key)
                else:
                    self._client = genai.Client()
            except ImportError:
                raise ImportError(
                    "Google Generative AI SDK not installed. "
                    "Install it with: pip install google-genai"
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
            **kwargs: Additional parameters (e.g., system_instruction, thinking_config)
            
        Returns:
            Generated text
        """
        self._initialize_client()
        
        # Build generation config
        from google.genai import types
        
        config_params = {**self.default_params}
        if temperature is not None:
            config_params['temperature'] = temperature
        if max_tokens is not None:
            config_params['max_output_tokens'] = max_tokens
        
        # Extract special parameters
        system_instruction = kwargs.pop('system_instruction', None)
        thinking_config = kwargs.pop('thinking_config', None)
        
        # Add any remaining kwargs
        config_params.update(kwargs)
        
        # Create config
        if system_instruction or thinking_config:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                thinking_config=thinking_config,
                **config_params
            )
        else:
            config = types.GenerateContentConfig(**config_params)
        
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"Google AI generation failed: {str(e)}")
    
    def get_provider_name(self) -> str:
        """Get provider name."""
        return f"Google AI Studio ({self.model_name})"
