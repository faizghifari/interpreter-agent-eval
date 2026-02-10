# Quick Start Guide

This guide will help you get started with the Interpreter Agent Evaluation Framework.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/faizghifari/interpreter-agent-eval.git
cd interpreter-agent-eval
```

2. Install the package:
```bash
pip install -e .
```

Or install with specific provider support:
```bash
pip install -e ".[openai]"     # For OpenAI
pip install -e ".[google]"     # For Google AI Studio
pip install -e ".[all]"        # For all providers
```

## Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:
```bash
OPENAI_API_KEY=your-key-here
# or
GOOGLE_API_KEY=your-key-here
```

## Basic Usage

### 1. Simple Example (No API Keys Required)

```python
from interpreter_agent_eval import User, InterpreterAgent, EvaluationFramework

# Create a mock provider for testing
class MockProvider:
    def generate(self, prompt, **kwargs):
        return "Hola, ¿cómo estás?"
    def get_provider_name(self):
        return "Mock"

# Create users
user1 = User(name="Alice", language="English", is_llm=False)
user2 = User(name="Carlos", language="Spanish", is_llm=False)

# Create interpreter
interpreter = InterpreterAgent(
    llm_provider=MockProvider(),
    translation_brief="Translate naturally",
    source_language="English",
    target_language="Spanish"
)

# Run evaluation
framework = EvaluationFramework(user1, user2, interpreter)
conversation = framework.run_conversation("Hello!", num_turns=3)

# Export results
framework.export_results('results.json', format='json')
```

### 2. Using OpenAI (Requires API Key)

```python
from interpreter_agent_eval import User, InterpreterAgent, EvaluationFramework
from interpreter_agent_eval.providers import OpenAIProvider
import os

# Create provider
provider = OpenAIProvider(
    api_key=os.getenv('OPENAI_API_KEY'),
    model_name='gpt-3.5-turbo'
)

# Create users and interpreter
user1 = User("Alice", "English", is_llm=False)
user2 = User("Carlos", "Spanish", is_llm=False)

interpreter = InterpreterAgent(
    llm_provider=provider,
    translation_brief="Translate naturally while preserving tone",
    source_language="English",
    target_language="Spanish"
)

# Run evaluation
framework = EvaluationFramework(user1, user2, interpreter)
conversation = framework.run_conversation(
    "I'm interested in learning about your culture.",
    num_turns=5
)
```

### 3. LLM-Powered Users

```python
from interpreter_agent_eval import User, InterpreterAgent, EvaluationFramework
from interpreter_agent_eval.providers import OpenAIProvider
import os

provider = OpenAIProvider(
    api_key=os.getenv('OPENAI_API_KEY'),
    model_name='gpt-4'
)

# Both users are LLM-powered
user1 = User(
    name="AI Alice",
    language="English",
    is_llm=True,
    llm_provider=provider,
    context="You are a friendly software developer from the US."
)

user2 = User(
    name="AI Carlos",
    language="Spanish",
    is_llm=True,
    llm_provider=provider,
    context="Eres un ingeniero de software amigable de España."
)

interpreter = InterpreterAgent(
    llm_provider=provider,
    translation_brief="Translate accurately",
    source_language="English",
    target_language="Spanish"
)

framework = EvaluationFramework(user1, user2, interpreter)
conversation = framework.run_conversation(
    "I'd like to discuss the project requirements.",
    num_turns=5
)
```

## Running Examples

The repository includes several example scripts:

```bash
# Basic usage examples
python examples/basic_usage.py

# Data handling demonstrations
python examples/data_handling_demo.py

# Advanced provider comparisons
python examples/advanced_usage.py
```

## Running Tests

```bash
python tests/test_basic.py
```

## Supported Providers

### OpenAI
```python
from interpreter_agent_eval.providers import OpenAIProvider
provider = OpenAIProvider(api_key="...", model_name="gpt-3.5-turbo")
```

### Google AI Studio
```python
from interpreter_agent_eval.providers import GoogleAIProvider
provider = GoogleAIProvider(api_key="...", model_name="gemini-pro")
```

### OpenRouter
```python
from interpreter_agent_eval.providers import OpenRouterProvider
provider = OpenRouterProvider(api_key="...", model_name="anthropic/claude-2")
```

### vLLM (Self-Hosted)
```python
from interpreter_agent_eval.providers import VLLMProvider
provider = VLLMProvider(
    base_url="http://localhost:8000",
    model_name="meta-llama/Llama-2-7b-chat-hf"
)
```

## Data Export

Export conversation results in multiple formats:

```python
# JSON format
framework.export_results('results.json', format='json')

# Text format
framework.export_results('results.txt', format='txt')

# CSV format
from interpreter_agent_eval.utils import DataHandler
DataHandler.export_to_csv(
    framework.conversation_log,
    'results.csv',
    fields=['turn', 'from_user', 'original_message', 'translated_message']
)
```

## Next Steps

- Explore the `examples/` directory for more use cases
- Check `config/` for example translation briefs and user contexts
- Read the full README.md for detailed documentation
- Customize translation briefs for your specific use case

## Getting Help

- Open an issue on GitHub
- Check the documentation in README.md
- Review the example scripts
