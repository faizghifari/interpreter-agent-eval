"""Advanced example: Using different LLM providers and comparing results."""

import os

from interpreter_agent_eval import User, InterpreterAgent, EvaluationFramework
from interpreter_agent_eval.providers import (
    OpenAIProvider,
    GoogleAIProvider,
    OpenRouterProvider,
)
from interpreter_agent_eval.utils import DataHandler


def compare_providers():
    """Compare different LLM providers for interpretation tasks."""

    # Define the common scenario
    translation_brief = DataHandler.load_translation_brief(
        os.path.join(
            os.path.dirname(__file__), "..", "config", "translation_brief_example.txt"
        )
    )

    initial_message = (
        "I need help understanding the technical documentation for your API."
    )

    providers_config = []

    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        providers_config.append(
            {
                "name": "OpenAI GPT-3.5",
                "provider": OpenAIProvider(
                    api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-3.5-turbo"
                ),
            }
        )

    # Google AI
    if os.getenv("GOOGLE_API_KEY"):
        providers_config.append(
            {
                "name": "Google Gemini",
                "provider": GoogleAIProvider(
                    api_key=os.getenv("GOOGLE_API_KEY"), model_name="gemini-pro"
                ),
            }
        )

    # OpenRouter
    if os.getenv("OPENROUTER_API_KEY"):
        providers_config.append(
            {
                "name": "OpenRouter Claude",
                "provider": OpenRouterProvider(
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    model_name="anthropic/claude-2",
                ),
            }
        )

    if not providers_config:
        print("No API keys configured. Please set at least one:")
        print("  OPENAI_API_KEY, GOOGLE_API_KEY, or OPENROUTER_API_KEY")
        return

    results = []

    for config in providers_config:
        print(f"\nTesting {config['name']}...")

        # Create users (manual for consistent testing, using ISO 639-3 codes)
        user1 = User(name="TechUser", language="eng", is_llm=False)  # English
        user2 = User(name="DevUser", language="spa", is_llm=False)  # Spanish

        # Create interpreter with this provider
        interpreter = InterpreterAgent(
            llm_provider=config["provider"],
            translation_brief=translation_brief,
            source_language="eng",
            target_language="spa",
            name=f"Interpreter-{config['name']}",
        )

        # Run evaluation with list of messages
        framework = EvaluationFramework(
            user1,
            user2,
            interpreter,
            name=f"comparison_{config['name'].replace(' ', '_')}",
        )

        messages = [
            "I need help understanding the technical documentation for your API.",
            "Of course! What specific part would you like me to explain?",
        ]
        conversation = framework.run_conversation(messages=messages)
        metrics = framework.evaluate_translation_quality()

        results.append(
            {
                "provider": config["name"],
                "metrics": metrics,
                "conversation": conversation,
            }
        )

        print(f"  Average translation time: {metrics['average_translation_time']:.3f}s")

    # Save comparison results
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    DataHandler.save_conversation_data(
        {"comparison_results": results},
        os.path.join(output_dir, "provider_comparison.json"),
    )

    print("\nComparison complete! Results saved to output/provider_comparison.json")

    # Print summary
    print("\nSummary:")
    print("-" * 60)
    for result in results:
        print(
            f"{result['provider']:20s} | "
            f"Avg Time: {result['metrics']['average_translation_time']:.3f}s"
        )


def multi_language_scenario():
    """Example with multiple language pairs."""

    print("\nMulti-Language Scenario Example")
    print("=" * 60)

    # Mock provider for demonstration
    class MockProvider:
        def generate(self, prompt, **kwargs):
            if "fra" in prompt or "French" in prompt:
                return "Bonjour! Comment puis-je vous aider?"
            elif "deu" in prompt or "German" in prompt:
                return "Guten Tag! Wie kann ich Ihnen helfen?"
            elif "jpn" in prompt or "Japanese" in prompt:
                return "こんにちは！どのようにお手伝いできますか？"
            return "Translation"

        def get_provider_name(self):
            return "Mock Multi-Language Provider"

    provider = MockProvider()

    # Test different language pairs (using ISO 639-3 codes)
    language_pairs = [
        ("eng", "fra"),  # English → French
        ("eng", "deu"),  # English → German
        ("eng", "jpn"),  # English → Japanese
    ]

    for source, target in language_pairs:
        print(f"\n{source} → {target}")

        user1 = User(name="User1", language=source, is_llm=False)
        user2 = User(name="User2", language=target, is_llm=False)

        interpreter = InterpreterAgent(
            llm_provider=provider,
            translation_brief="Translate accurately between languages",
            source_language=source,
            target_language=target,
        )

        framework = EvaluationFramework(user1, user2, interpreter)
        conversation = framework.run_conversation(
            messages=["Hello! How can I help you?"]
        )

        print(f"  Original: {conversation[0]['original_message']}")
        print(f"  Translated: {conversation[0]['translated_message']}")


if __name__ == "__main__":
    print("Advanced Examples - Provider Comparison and Multi-Language\n")

    compare_providers()
    multi_language_scenario()

    print("\n" + "=" * 60)
    print("Advanced examples completed!")
    print("=" * 60)
