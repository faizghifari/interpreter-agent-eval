"""Evaluation framework for interpreter agents."""

from typing import List, Dict, Any, Optional
import json
import time
from datetime import datetime
from pathlib import Path

from .user import User
from .interpreter import InterpreterAgent


class EvaluationFramework:
    """Framework for evaluating interpreter/translator agents.
    
    Orchestrates conversations between two users (potentially LLM-powered)
    speaking different languages, with an interpreter agent facilitating communication.
    """
    
    def __init__(
        self,
        user1: User,
        user2: User,
        interpreter: InterpreterAgent,
        name: Optional[str] = None
    ):
        """Initialize the evaluation framework.
        
        Args:
            user1: First user
            user2: Second user
            interpreter: Interpreter agent
            name: Optional name for this evaluation session
        """
        self.user1 = user1
        self.user2 = user2
        self.interpreter = interpreter
        self.name = name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversation_log = []
        self.metrics = {}
    
    def run_conversation(
        self,
        messages: List[str],
        from_user: int = 1
    ) -> List[Dict[str, Any]]:
        """Run a conversation between the two users via the interpreter.
        
        Args:
            messages: List of messages to exchange. Each message is sent by alternating users.
            from_user: Which user starts (1 or 2)
            
        Returns:
            List of conversation exchanges
            
        Note:
            Users only communicate through the interpreter agent. Each user's conversation
            history contains only their own messages and the interpreter's translated responses.
            For LLM-powered users, they generate responses based on the translated messages
            they receive from the interpreter.
        """
        current_user = self.user1 if from_user == 1 else self.user2
        other_user = self.user2 if from_user == 1 else self.user1
        
        for turn, message in enumerate(messages):
            # Record the turn
            turn_data = {
                "turn": turn + 1,
                "timestamp": datetime.now().isoformat(),
                "from_user": current_user.name,
                "to_user": other_user.name
            }
            
            # Current user sends message in their language
            sent_message = current_user.send_message(message)
            turn_data["original_message"] = sent_message
            turn_data["original_language"] = current_user.language
            
            # Interpreter translates
            start_time = time.time()
            translation_result = self.interpreter.facilitate_conversation(
                sent_message,
                current_user.language,
                other_user.language,
                context=f"Turn {turn + 1} of conversation"
            )
            translation_time = time.time() - start_time
            
            turn_data["translated_message"] = translation_result["translation"]
            turn_data["translated_language"] = translation_result["translation_language"]
            turn_data["translation_time"] = translation_time
            
            # Other user receives the translation from interpreter (not directly from the user)
            other_user.receive_message(translation_result["translation"], metadata={"from": "interpreter"})
            
            # Log the turn
            self.conversation_log.append(turn_data)
            
            # Swap users for next turn
            current_user, other_user = other_user, current_user
        
        return self.conversation_log
    
    def evaluate_translation_quality(self) -> Dict[str, Any]:
        """Evaluate the quality of translations.
        
        This is a placeholder for various evaluation metrics.
        In practice, you might use BLEU, COMET, or other MT metrics.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.conversation_log:
            return {"error": "No conversation data to evaluate"}
        
        metrics = {
            "total_turns": len(self.conversation_log),
            "average_translation_time": sum(
                turn.get("translation_time", 0) for turn in self.conversation_log
            ) / len(self.conversation_log),
            "languages": {
                self.user1.language: self.user1.name,
                self.user2.language: self.user2.name
            }
        }
        
        self.metrics = metrics
        return metrics
    
    def export_results(self, filepath: str, format: str = "json") -> None:
        """Export evaluation results to a file.
        
        Args:
            filepath: Path to save the results
            format: Export format ('json' or 'txt')
        """
        results = {
            "session_name": self.name,
            "timestamp": datetime.now().isoformat(),
            "users": {
                "user1": {
                    "name": self.user1.name,
                    "language": self.user1.language,
                    "is_llm": self.user1.is_llm
                },
                "user2": {
                    "name": self.user2.name,
                    "language": self.user2.language,
                    "is_llm": self.user2.is_llm
                }
            },
            "interpreter": {
                "name": self.interpreter.name,
                "translation_brief": self.interpreter.translation_brief
            },
            "conversation": self.conversation_log,
            "metrics": self.metrics
        }
        
        # Ensure parent directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        elif format == "txt":
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Evaluation Session: {self.name}\n")
                f.write(f"Timestamp: {results['timestamp']}\n\n")
                f.write(f"Users:\n")
                f.write(f"  User 1: {self.user1.name} ({self.user1.language})\n")
                f.write(f"  User 2: {self.user2.name} ({self.user2.language})\n\n")
                f.write(f"Interpreter: {self.interpreter.name}\n\n")
                f.write("Conversation:\n")
                f.write("=" * 80 + "\n")
                for turn in self.conversation_log:
                    f.write(f"\nTurn {turn['turn']}:\n")
                    f.write(f"  From: {turn['from_user']} ({turn['original_language']})\n")
                    f.write(f"  Message: {turn['original_message']}\n")
                    f.write(f"  Translation: {turn['translated_message']}\n")
                    f.write(f"  Time: {turn['translation_time']:.3f}s\n")
                f.write("\n" + "=" * 80 + "\n")
                f.write("\nMetrics:\n")
                for key, value in self.metrics.items():
                    f.write(f"  {key}: {value}\n")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation.
        
        Returns:
            Summary dictionary
        """
        return {
            "session_name": self.name,
            "total_turns": len(self.conversation_log),
            "user1": f"{self.user1.name} ({self.user1.language})",
            "user2": f"{self.user2.name} ({self.user2.language})",
            "interpreter": self.interpreter.name,
            "metrics": self.metrics
        }
