"""Example demonstrating data handling and aggregation utilities."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from interpreter_agent_eval import User, InterpreterAgent, EvaluationFramework
from interpreter_agent_eval.utils import DataHandler


def demonstrate_data_export():
    """Show different data export formats."""
    
    print("=" * 80)
    print("Data Handling Examples")
    print("=" * 80)
    
    # Mock provider
    class MockProvider:
        def generate(self, prompt, **kwargs):
            return "Translated message"
        def get_provider_name(self):
            return "Mock Provider"
    
    # Create a sample conversation
    user1 = User("Alice", "eng", is_llm=False)  # English
    user2 = User("Bob", "spa", is_llm=False)    # Spanish
    interpreter = InterpreterAgent(
        llm_provider=MockProvider(),
        translation_brief="Translate accurately",
        source_language="eng",
        target_language="spa"
    )
    
    framework = EvaluationFramework(user1, user2, interpreter, name="data_demo")
    conversation = framework.run_conversation(messages=["Hello there!", "Hi! How are you?", "I'm good, thanks!"])
    framework.evaluate_translation_quality()
    
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Export as JSON
    print("\n1. Exporting as JSON...")
    json_path = os.path.join(output_dir, 'data_demo.json')
    framework.export_results(json_path, format='json')
    print(f"   Saved to: {json_path}")
    
    # 2. Export as TXT
    print("\n2. Exporting as TXT...")
    txt_path = os.path.join(output_dir, 'data_demo.txt')
    framework.export_results(txt_path, format='txt')
    print(f"   Saved to: {txt_path}")
    
    # 3. Export to CSV
    print("\n3. Exporting to CSV...")
    csv_path = os.path.join(output_dir, 'data_demo.csv')
    DataHandler.export_to_csv(
        conversation,
        csv_path,
        fields=['turn', 'from_user', 'original_message', 'translated_message', 'translation_time']
    )
    print(f"   Saved to: {csv_path}")
    
    # 4. Load and display JSON data
    print("\n4. Loading JSON data back...")
    loaded_data = DataHandler.load_conversation_data(json_path)
    print(f"   Session name: {loaded_data['session_name']}")
    print(f"   Total turns: {loaded_data['metrics']['total_turns']}")


def demonstrate_aggregation():
    """Show how to aggregate results from multiple evaluations."""
    
    print("\n" + "=" * 80)
    print("Data Aggregation Examples")
    print("=" * 80)
    
    # Mock provider
    class MockProvider:
        def generate(self, prompt, **kwargs):
            return "Translated"
        def get_provider_name(self):
            return "Mock Provider"
    
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    result_files = []
    
    # Create multiple evaluation runs
    for i in range(3):
        user1 = User(f"User{i}_1", "eng", is_llm=False)  # English
        user2 = User(f"User{i}_2", "spa", is_llm=False)  # Spanish
        interpreter = InterpreterAgent(
            llm_provider=MockProvider(),
            translation_brief="Translate",
            source_language="eng",
            target_language="spa"
        )
        
        framework = EvaluationFramework(user1, user2, interpreter, name=f"eval_{i}")
        num_messages = 2 + i
        messages = [f"Test message {j+1}" for j in range(num_messages)]
        framework.run_conversation(messages=messages)
        metrics = framework.evaluate_translation_quality()
        
        filepath = os.path.join(output_dir, f'eval_{i}.json')
        framework.export_results(filepath, format='json')
        result_files.append(filepath)
        print(f"\n Created evaluation {i}: {metrics['total_turns']} turns")
    
    # Aggregate results
    print("\n Aggregating results...")
    aggregated = DataHandler.aggregate_results(result_files)
    
    print(f"\nAggregation Summary:")
    print(f"  Number of evaluations: {aggregated['num_evaluations']}")
    print(f"  Total turns across all: {aggregated['total_turns']}")
    print(f"  Average translation time: {aggregated['average_translation_time']:.6f}s")
    
    # Save aggregated results
    agg_path = os.path.join(output_dir, 'aggregated_results.json')
    DataHandler.save_conversation_data(aggregated, agg_path)
    print(f"\n Aggregated results saved to: {agg_path}")


def demonstrate_context_loading():
    """Show how to load context from files."""
    
    print("\n" + "=" * 80)
    print("Context Loading Examples")
    print("=" * 80)
    
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
    
    # Load translation brief
    print("\n1. Loading translation brief...")
    brief_path = os.path.join(config_dir, 'translation_brief_example.txt')
    if os.path.exists(brief_path):
        brief = DataHandler.load_translation_brief(brief_path)
        print(f"   Loaded {len(brief)} characters")
        print(f"   Preview: {brief[:100]}...")
    
    # Load user contexts
    print("\n2. Loading user contexts...")
    for lang in ['en', 'es']:
        context_path = os.path.join(config_dir, f'user_context_{lang}_example.txt')
        if os.path.exists(context_path):
            context = DataHandler.load_user_context(context_path)
            print(f"   Loaded {lang.upper()} context: {len(context)} characters")


if __name__ == "__main__":
    print("Data Handling and Utilities Examples\n")
    
    demonstrate_data_export()
    demonstrate_aggregation()
    demonstrate_context_loading()
    
    print("\n" + "=" * 80)
    print("Data handling examples completed!")
    print("=" * 80)
