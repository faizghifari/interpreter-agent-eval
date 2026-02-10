"""Interpreter agent for translating between users."""

from typing import Optional, Dict, Any


class InterpreterAgent:
    """Interpreter/translator agent that bridges communication between users.
    
    The interpreter receives a translation brief and facilitates communication
    between users speaking different languages.
    """
    
    def __init__(
        self,
        llm_provider: Any,
        translation_brief: str,
        source_language: str,
        target_language: str,
        name: str = "Interpreter"
    ):
        """Initialize the InterpreterAgent.
        
        Args:
            llm_provider: LLM provider instance for translation
            translation_brief: Instructions/guidelines for translation
            source_language: Source language code
            target_language: Target language code
            name: Name for the interpreter agent
        """
        self.llm_provider = llm_provider
        self.translation_brief = translation_brief
        self.source_language = source_language
        self.target_language = target_language
        self.name = name
        self.translation_history = []
    
    def translate(
        self,
        message: str,
        from_language: Optional[str] = None,
        to_language: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        """Translate a message from one language to another.
        
        Args:
            message: The message to translate
            from_language: Source language (uses source_language if not provided)
            to_language: Target language (uses target_language if not provided)
            context: Additional context for translation
            
        Returns:
            Translated message
        """
        from_lang = from_language or self.source_language
        to_lang = to_language or self.target_language
        
        prompt = self._build_translation_prompt(message, from_lang, to_lang, context)
        translation = self.llm_provider.generate(prompt)
        
        # Record translation
        self.translation_history.append({
            "original": message,
            "translation": translation,
            "from": from_lang,
            "to": to_lang,
            "context": context
        })
        
        return translation
    
    def _build_translation_prompt(
        self,
        message: str,
        from_language: str,
        to_language: str,
        context: Optional[str] = None
    ) -> str:
        """Build a translation prompt for the LLM.
        
        Args:
            message: Message to translate
            from_language: Source language
            to_language: Target language
            context: Optional context
            
        Returns:
            Formatted translation prompt
        """
        prompt_parts = [
            f"Translation Brief: {self.translation_brief}",
            "",
            f"Translate the following message from {from_language} to {to_language}."
        ]
        
        if context:
            prompt_parts.append(f"Context: {context}")
        
        prompt_parts.extend([
            "",
            f"Message to translate: {message}",
            "",
            f"Translation ({to_language}):"
        ])
        
        return "\n".join(prompt_parts)
    
    def get_translation_history(self) -> list:
        """Get the translation history.
        
        Returns:
            List of translation records
        """
        return self.translation_history
    
    def facilitate_conversation(
        self,
        message: str,
        sender_language: str,
        receiver_language: str,
        context: Optional[str] = None
    ) -> Dict[str, str]:
        """Facilitate a bidirectional conversation.
        
        Args:
            message: Message from sender
            sender_language: Language of the sender
            receiver_language: Language of the receiver
            context: Optional context
            
        Returns:
            Dictionary with original and translated messages
        """
        translation = self.translate(
            message,
            from_language=sender_language,
            to_language=receiver_language,
            context=context
        )
        
        return {
            "original": message,
            "original_language": sender_language,
            "translation": translation,
            "translation_language": receiver_language
        }
