"""Basic tests for the interpreter agent evaluation framework."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from interpreter_agent_eval import User, InterpreterAgent, EvaluationFramework
from interpreter_agent_eval.providers.base import LLMProvider
from interpreter_agent_eval.utils import DataHandler


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, response="Mock response"):
        self.response = response
        self.call_count = 0
    
    def generate(self, prompt, max_tokens=None, temperature=None, **kwargs):
        self.call_count += 1
        return self.response
    
    def get_provider_name(self):
        return "Mock Provider"


def test_user_creation():
    """Test creating users."""
    print("Testing user creation...")
    
    # Test manual user
    user1 = User(name="Alice", language="English", is_llm=False)
    assert user1.name == "Alice"
    assert user1.language == "English"
    assert user1.is_llm is False
    assert len(user1.conversation_history) == 0
    
    # Test LLM user
    provider = MockLLMProvider()
    user2 = User(name="Bob", language="Spanish", is_llm=True, llm_provider=provider)
    assert user2.is_llm is True
    assert user2.llm_provider is not None
    
    print("  ✓ User creation tests passed")


def test_user_send_message():
    """Test user sending messages."""
    print("Testing user message sending...")
    
    # Manual user
    user1 = User(name="Alice", language="English", is_llm=False)
    msg = user1.send_message("Hello")
    assert msg == "Hello"
    assert len(user1.conversation_history) == 1
    
    # LLM user
    provider = MockLLMProvider(response="Hola")
    user2 = User(name="Bob", language="Spanish", is_llm=True, llm_provider=provider)
    msg = user2.send_message("Input message")
    assert msg == "Hola"
    assert provider.call_count == 1
    
    print("  ✓ Message sending tests passed")


def test_interpreter_translation():
    """Test interpreter translation functionality."""
    print("Testing interpreter translation...")
    
    provider = MockLLMProvider(response="Hola, ¿cómo estás?")
    interpreter = InterpreterAgent(
        llm_provider=provider,
        translation_brief="Translate accurately",
        source_language="English",
        target_language="Spanish"
    )
    
    translation = interpreter.translate("Hello, how are you?")
    assert translation == "Hola, ¿cómo estás?"
    assert len(interpreter.translation_history) == 1
    assert provider.call_count == 1
    
    # Test history
    history = interpreter.get_translation_history()
    assert len(history) == 1
    assert history[0]["original"] == "Hello, how are you?"
    assert history[0]["from"] == "English"
    assert history[0]["to"] == "Spanish"
    
    print("  ✓ Interpreter translation tests passed")


def test_evaluation_framework():
    """Test evaluation framework orchestration."""
    print("Testing evaluation framework...")
    
    # Setup
    provider = MockLLMProvider(response="Translation")
    user1 = User(name="Alice", language="English", is_llm=False)
    user2 = User(name="Bob", language="Spanish", is_llm=False)
    interpreter = InterpreterAgent(
        llm_provider=provider,
        translation_brief="Translate",
        source_language="English",
        target_language="Spanish"
    )
    
    # Create framework
    framework = EvaluationFramework(user1, user2, interpreter, name="test_eval")
    assert framework.user1 == user1
    assert framework.user2 == user2
    assert framework.interpreter == interpreter
    
    # Run conversation
    conversation = framework.run_conversation("Hello", num_turns=3)
    assert len(conversation) == 3
    assert all('turn' in turn for turn in conversation)
    assert all('original_message' in turn for turn in conversation)
    assert all('translated_message' in turn for turn in conversation)
    
    # Evaluate
    metrics = framework.evaluate_translation_quality()
    assert 'total_turns' in metrics
    assert metrics['total_turns'] == 3
    assert 'average_translation_time' in metrics
    
    print("  ✓ Evaluation framework tests passed")


def test_data_handler():
    """Test data handling utilities."""
    print("Testing data handler...")
    
    import tempfile
    import json
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test save and load
        test_data = {
            "session_name": "test",
            "metrics": {"total_turns": 5}
        }
        
        filepath = os.path.join(tmpdir, "test.json")
        DataHandler.save_conversation_data(test_data, filepath)
        
        loaded_data = DataHandler.load_conversation_data(filepath)
        assert loaded_data["session_name"] == "test"
        assert loaded_data["metrics"]["total_turns"] == 5
        
        # Test CSV export
        conversation_log = [
            {"turn": 1, "from_user": "Alice", "message": "Hello"},
            {"turn": 2, "from_user": "Bob", "message": "Hi"}
        ]
        
        csv_path = os.path.join(tmpdir, "test.csv")
        DataHandler.export_to_csv(conversation_log, csv_path)
        assert os.path.exists(csv_path)
    
    print("  ✓ Data handler tests passed")


def test_mock_provider():
    """Test the mock provider interface."""
    print("Testing mock provider...")
    
    provider = MockLLMProvider(response="Test response")
    
    result = provider.generate("Test prompt")
    assert result == "Test response"
    assert provider.call_count == 1
    
    result = provider.generate("Another prompt", max_tokens=100)
    assert result == "Test response"
    assert provider.call_count == 2
    
    name = provider.get_provider_name()
    assert name == "Mock Provider"
    
    print("  ✓ Mock provider tests passed")


def run_all_tests():
    """Run all test functions."""
    print("\n" + "=" * 80)
    print("Running Interpreter Agent Evaluation Framework Tests")
    print("=" * 80 + "\n")
    
    tests = [
        test_user_creation,
        test_user_send_message,
        test_interpreter_translation,
        test_evaluation_framework,
        test_data_handler,
        test_mock_provider
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} failed: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
