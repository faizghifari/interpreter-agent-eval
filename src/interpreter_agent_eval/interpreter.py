"""Interpreter agent for translating between users."""

from typing import Optional, Dict, Any
from iso639 import Lang
from interpreter_agent_eval.prompts.templates import (
    DEFAULT_TRANSLATION_BRIEF,
    TRANSLATION_TASK,
)


class InterpreterAgent:
    """Interpreter/translator agent that bridges communication between users.

    The interpreter receives a translation brief and facilitates communication
    between users speaking different languages.
    """

    def __init__(
        self,
        llm_provider: Any,
        source_language: str,
        target_language: str,
        translation_brief: Optional[str] = None,
        conversation_context: Optional[str] = None,
        name: str = "Interpreter",
    ):
        """Initialize the InterpreterAgent.

        Args:
            llm_provider: LLM provider instance for translation
            source_language: Source language code (ISO 639-3)
            target_language: Target language code (ISO 639-3)
            translation_brief: Optional custom instructions/guidelines for translation.
                If None, uses DEFAULT_TRANSLATION_BRIEF populated with source/target languages and conversation_context.
            conversation_context: Optional brief description of the conversation scenario.
                If None, uses a generic placeholder.
            name: Name for the interpreter agent
        """
        self.llm_provider = llm_provider
        self.source_language = source_language
        self.target_language = target_language
        self.conversation_context = (
            conversation_context or "A general conversation between two users."
        )
        self.name = name
        self.translation_history = []

        if translation_brief:
            self.translation_brief = translation_brief
        else:
            self.translation_brief = self._prepare_default_brief(
                source_language, target_language, self.conversation_context
            )

    def _prepare_default_brief(
        self, source_code: str, target_code: str, conversation_context: str
    ) -> str:
        """Prepare the default translation brief with language details and conversation context.

        Args:
            source_code: Source language code
            target_code: Target language code
            conversation_context: Brief description of the conversation scenario

        Returns:
            Populated translation brief
        """
        try:
            source_lang = Lang(source_code)
            source_name = f"{source_lang.name} ({source_lang.pt3})"
        except Exception:
            source_name = source_code

        try:
            target_lang = Lang(target_code)
            target_name = f"{target_lang.name} ({target_lang.pt3})"
        except Exception:
            target_name = target_code

        return DEFAULT_TRANSLATION_BRIEF.format(
            user_a_language=source_name,
            user_b_language=target_name,
            conversation_context=conversation_context,
        )

    def translate(
        self,
        message: str,
        from_language: Optional[str] = None,
        to_language: Optional[str] = None,
        context: Optional[str] = None,
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
        translation = self.llm_provider.generate(
            prompt, system_prompt=self.translation_brief
        )

        # Record translation
        self.translation_history.append(
            {
                "original": message,
                "translation": translation,
                "from": from_lang,
                "to": to_lang,
                "context": context,
            }
        )

        return translation

    def _build_translation_prompt(
        self,
        message: str,
        from_language: str,
        to_language: str,
        context: Optional[str] = None,
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
        # Get language names for the prompt
        try:
            from_lang_val = Lang(from_language)
            from_lang_name = f"{from_lang_val.name} ({from_lang_val.pt3})"
        except Exception:
            from_lang_name = from_language

        try:
            to_lang_val = Lang(to_language)
            to_lang_name = f"{to_lang_val.name} ({to_lang_val.pt3})"
        except Exception:
            to_lang_name = to_language

        context_str = f"Context: {context}" if context else ""

        return TRANSLATION_TASK.format(
            from_language=from_lang_name,
            to_language=to_lang_name,
            context=context_str,
            message=message,
        )

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
        context: Optional[str] = None,
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
            context=context,
        )

        return {
            "original": message,
            "original_language": sender_language,
            "translation": translation,
            "translation_language": receiver_language,
        }
