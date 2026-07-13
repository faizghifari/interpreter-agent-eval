"""User class for representing conversation participants."""

from typing import Optional, Dict, Any


class User:
    """Represents a user in the conversation with a specific language.

    Users can be either human users or LLM-powered users.
    """

    def __init__(
        self,
        name: str,
        language: str,
        is_llm: bool = False,
        llm_provider: Optional[Any] = None,
        context: Optional[str] = None,
        language_name: Optional[str] = None,
    ):
        """Initialize a User.

        Args:
            name: Name/identifier for the user
            language: ISO 639-3 language code (e.g., 'eng' for English, 'spa' for Spanish)
            is_llm: Whether this user is powered by an LLM
            llm_provider: LLM provider instance if is_llm is True
            context: System-level context and constraints for the user (injected as system prompt)
            language_name: Human-readable language name (e.g., 'Arabic'). Falls back to language code.
        """
        self.name = name
        self.language = language
        self.language_name = language_name or language
        self.is_llm = is_llm
        self.llm_provider = llm_provider
        self.context = context
        self.conversation_history = []

    def send_message(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Send a message from this user.

        For LLM users: `message` is the incoming text to respond to (e.g. the interpreter's
        translation). It goes as the user-role turn; context/constraints go as the system prompt.

        Args:
            message: The message content to respond to (for LLM users) or to record (for humans)
            metadata: Optional metadata about the message

        Returns:
            The message sent (human) or the LLM-generated response
        """
        if self.is_llm and self.llm_provider:
            system_prompt = self._build_system_prompt()
            response = self.llm_provider.generate(message, system_prompt=system_prompt)
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "metadata": metadata,
            })
            return response
        else:
            self.conversation_history.append({
                "role": "user",
                "content": message,
                "metadata": metadata,
            })
            return message

    def receive_message(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Record a received message in conversation history (for logging purposes)."""
        self.conversation_history.append({
            "role": "received",
            "content": message,
            "metadata": metadata,
        })

    def _build_system_prompt(self) -> Optional[str]:
        """Build the system prompt: context and constraints only."""
        if self.context:
            return self.context
        return None
        return "\n\n".join(parts)

    def get_conversation_history(self) -> list:
        return self.conversation_history
