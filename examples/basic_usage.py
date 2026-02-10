"""Example usage of the interpreter agent evaluation framework."""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from interpreter_agent_eval import User, InterpreterAgent, EvaluationFramework

# This example demonstrates basic usage with mock/manual responses
# For real usage, you would configure actual API keys


def example_with_manual_users():
    """Example with manual (non-LLM) users."""
    print("=" * 80)
    print("Example 1: Manual Users")
    print("=" * 80)
    
    # Create a simple mock provider for demonstration
    class MockProvider:
        def generate(self, prompt, **kwargs):
            # Simple mock translation
            if "Spanish" in prompt:
                return "Hola, ¿cómo estás?"
            elif "English" in prompt:
                return "Hello, how are you?"
            return "Translation result"
        
        def get_provider_name(self):
            return "Mock Provider"
    
    # Create two users speaking different languages
    user1 = User(
        name="Alice",
        language="English",
        is_llm=False,
        context="You are a friendly person having a casual conversation"
    )
    
    user2 = User(
        name="Carlos",
        language="Spanish",
        is_llm=False,
        context="Eres una persona amigable teniendo una conversación casual"
    )
    
    # Create interpreter with translation brief
    interpreter = InterpreterAgent(
        llm_provider=MockProvider(),
        translation_brief="Translate naturally and maintain the conversational tone.",
        source_language="English",
        target_language="Spanish",
        name="TranslatorBot"
    )
    
    # Create evaluation framework
    framework = EvaluationFramework(
        user1=user1,
        user2=user2,
        interpreter=interpreter,
        name="example_manual_users"
    )
    
    # Run a simple conversation
    print("\nStarting conversation...")
    initial_message = "Hello! How are you doing today?"
    conversation = framework.run_conversation(
        initial_message=initial_message,
        num_turns=3,
        from_user=1
    )
    
    # Display conversation
    print("\nConversation Log:")
    for turn in conversation:
        print(f"\nTurn {turn['turn']}:")
        print(f"  {turn['from_user']}: {turn['original_message']} ({turn['original_language']})")
        print(f"  Translation: {turn['translated_message']} ({turn['translated_language']})")
        print(f"  Time: {turn['translation_time']:.3f}s")
    
    # Evaluate
    metrics = framework.evaluate_translation_quality()
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Export results
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    json_path = os.path.join(output_dir, 'example_manual_users.json')
    framework.export_results(json_path, format='json')
    print(f"\nResults exported to: {json_path}")
    
    txt_path = os.path.join(output_dir, 'example_manual_users.txt')
    framework.export_results(txt_path, format='txt')
    print(f"Results exported to: {txt_path}")


def example_with_llm_users():
    """Example with LLM-powered users (requires API keys)."""
    print("\n" + "=" * 80)
    print("Example 2: LLM-Powered Users (Requires API Keys)")
    print("=" * 80)
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("\nSkipping LLM example - OPENAI_API_KEY not set")
        print("To run this example, set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        return
    
    print("\nNote: This example uses real API calls and will incur costs.")
    print("Uncomment the code below to run.")
    
    # Uncomment to actually run with real API
    """
    # Create LLM provider
    llm_provider = OpenAIProvider(
        api_key=api_key,
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Create LLM-powered users
    user1 = User(
        name="AI Alice",
        language="English",
        is_llm=True,
        llm_provider=llm_provider,
        context="You are a friendly English speaker discussing travel plans"
    )
    
    user2 = User(
        name="AI Carlos",
        language="Spanish",
        is_llm=True,
        llm_provider=llm_provider,
        context="Eres un hablante de español amigable discutiendo planes de viaje"
    )
    
    # Create interpreter
    interpreter = InterpreterAgent(
        llm_provider=llm_provider,
        translation_brief="Translate accurately while preserving tone and context.",
        source_language="English",
        target_language="Spanish",
        name="AI Translator"
    )
    
    # Create evaluation framework
    framework = EvaluationFramework(
        user1=user1,
        user2=user2,
        interpreter=interpreter,
        name="example_llm_users"
    )
    
    # Run conversation
    conversation = framework.run_conversation(
        initial_message="I'm thinking about visiting Barcelona next month.",
        num_turns=3,
        from_user=1
    )
    
    # Evaluate and export
    metrics = framework.evaluate_translation_quality()
    
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    framework.export_results(
        os.path.join(output_dir, 'example_llm_users.json'),
        format='json'
    )
    print("Results exported successfully")
    """


def example_google_ai():
    """Example using Google AI Studio."""
    print("\n" + "=" * 80)
    print("Example 3: Google AI Studio Provider")
    print("=" * 80)
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("\nSkipping Google AI example - GOOGLE_API_KEY not set")
        print("To run this example, set your Google AI Studio API key:")
        print("  export GOOGLE_API_KEY='your-api-key-here'")
        return
    
    print("\nGoogle AI Studio provider is configured.")
    print("Uncomment the code in the example to use it.")


def example_vllm():
    """Example using vLLM server."""
    print("\n" + "=" * 80)
    print("Example 4: vLLM Server")
    print("=" * 80)
    
    print("\nTo use vLLM:")
    print("1. Start a vLLM server with a HuggingFace model:")
    print("   python -m vllm.entrypoints.openai.api_server \\")
    print("     --model meta-llama/Llama-2-7b-chat-hf \\")
    print("     --port 8000")
    print("2. Configure the VLLMProvider with the server URL")
    print("3. Uncomment the code in the example to use it")


if __name__ == "__main__":
    print("Interpreter Agent Evaluation Framework - Examples\n")
    
    # Run examples
    example_with_manual_users()
    example_with_llm_users()
    example_google_ai()
    example_vllm()
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
