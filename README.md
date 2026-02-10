# Interpreter Agent Evaluation Framework

A comprehensive evaluation framework for interpreter/translator agents that facilitate communication between users speaking different languages.

## Overview

This framework enables evaluation of LLM-powered interpreter agents that bridge communication between two users speaking different languages. The framework supports:

- **Multiple LLM Providers**: Google AI Studio, OpenAI, OpenRouter, and vLLM (HuggingFace models)
- **Flexible User Types**: Human users or LLM-powered users
- **Translation Evaluation**: Track and analyze translation quality and performance
- **Data Management**: Export and analyze conversation data

## Features

### Core Components

1. **User Class**: Represents conversation participants with language preferences
   - Support for both human and LLM-powered users
   - Conversation history tracking
   - Context management

2. **Interpreter Agent**: LLM-powered translator that facilitates communication
   - Receives translation briefs with guidelines
   - Translates between arbitrary language pairs
   - Tracks translation history

3. **Evaluation Framework**: Orchestrates conversations and collects metrics
   - Run multi-turn conversations
   - Calculate performance metrics
   - Export results in multiple formats

### Supported LLM Providers

- **Google AI Studio** (`GoogleAIProvider`): Gemini models via Google Generative AI SDK
- **OpenAI** (`OpenAIProvider`): GPT models via OpenAI API
- **OpenRouter** (`OpenRouterProvider`): Access to various models through OpenRouter
- **vLLM** (`VLLMProvider`): Self-hosted HuggingFace models via vLLM server

## Installation

1. Clone the repository:
```bash
git clone https://github.com/faizghifari/interpreter-agent-eval.git
cd interpreter-agent-eval
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys (choose providers you need):
```bash
# For OpenAI
export OPENAI_API_KEY='your-openai-api-key'

# For Google AI Studio
export GOOGLE_API_KEY='your-google-api-key'

# For OpenRouter
export OPENROUTER_API_KEY='your-openrouter-api-key'
```

## Quick Start

### Basic Example with Manual Users

```python
from interpreter_agent_eval import User, InterpreterAgent, EvaluationFramework
from interpreter_agent_eval.providers import OpenAIProvider

# Create users
user1 = User(name="Alice", language="English", is_llm=False)
user2 = User(name="Carlos", language="Spanish", is_llm=False)

# Create interpreter with an LLM provider
provider = OpenAIProvider(api_key="your-key", model_name="gpt-3.5-turbo")
interpreter = InterpreterAgent(
    llm_provider=provider,
    translation_brief="Translate naturally and maintain conversational tone.",
    source_language="English",
    target_language="Spanish"
)

# Create evaluation framework
framework = EvaluationFramework(user1, user2, interpreter)

# Run conversation
conversation = framework.run_conversation(
    initial_message="Hello! How are you?",
    num_turns=5
)

# Evaluate and export results
metrics = framework.evaluate_translation_quality()
framework.export_results('results.json', format='json')
```

### Example with LLM-Powered Users

```python
from interpreter_agent_eval import User, InterpreterAgent, EvaluationFramework
from interpreter_agent_eval.providers import OpenAIProvider

# Create LLM provider
provider = OpenAIProvider(api_key="your-key", model_name="gpt-4")

# Create LLM-powered users
user1 = User(
    name="AI Alice",
    language="English",
    is_llm=True,
    llm_provider=provider,
    context="You are a friendly professional discussing a project."
)

user2 = User(
    name="AI Carlos", 
    language="Spanish",
    is_llm=True,
    llm_provider=provider,
    context="Eres un profesional amigable discutiendo un proyecto."
)

# Create interpreter
interpreter = InterpreterAgent(
    llm_provider=provider,
    translation_brief="Translate accurately while preserving context.",
    source_language="English",
    target_language="Spanish"
)

# Run evaluation
framework = EvaluationFramework(user1, user2, interpreter)
conversation = framework.run_conversation("Let's discuss the project timeline.", num_turns=5)
```

### Using Different Providers

#### Google AI Studio
```python
from interpreter_agent_eval.providers import GoogleAIProvider

provider = GoogleAIProvider(
    api_key="your-google-api-key",
    model_name="gemini-pro",
    temperature=0.7
)
```

#### OpenRouter
```python
from interpreter_agent_eval.providers import OpenRouterProvider

provider = OpenRouterProvider(
    api_key="your-openrouter-key",
    model_name="anthropic/claude-2",
    site_url="https://yourapp.com",
    app_name="Your App"
)
```

#### vLLM (Self-Hosted)
```python
from interpreter_agent_eval.providers import VLLMProvider

# First, start vLLM server:
# python -m vllm.entrypoints.openai.api_server \
#   --model meta-llama/Llama-2-7b-chat-hf \
#   --port 8000

provider = VLLMProvider(
    base_url="http://localhost:8000",
    model_name="meta-llama/Llama-2-7b-chat-hf"
)
```

## Usage Examples

Run the example script:
```bash
python examples/basic_usage.py
```

This will demonstrate:
- Manual users with mock translations
- LLM-powered users (if API keys are set)
- Different provider configurations

## Project Structure

```
interpreter-agent-eval/
├── src/
│   └── interpreter_agent_eval/
│       ├── __init__.py           # Main package exports
│       ├── user.py               # User class
│       ├── interpreter.py        # Interpreter agent
│       ├── evaluator.py          # Evaluation framework
│       ├── providers/            # LLM provider implementations
│       │   ├── __init__.py
│       │   ├── base.py           # Base provider class
│       │   ├── google_ai.py      # Google AI Studio
│       │   ├── openai.py         # OpenAI
│       │   ├── openrouter.py     # OpenRouter
│       │   └── vllm.py           # vLLM
│       └── utils/                # Utility functions
│           ├── __init__.py
│           └── data_handler.py   # Data handling utilities
├── examples/
│   └── basic_usage.py            # Example usage scripts
├── config/
│   ├── config.template           # Configuration template
│   ├── translation_brief_example.txt
│   └── user_context_*.txt        # Example contexts
├── tests/                        # Test files (to be added)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── LICENSE

```

## Data Handling

The framework includes utilities for data management:

```python
from interpreter_agent_eval.utils import DataHandler

# Export to CSV
DataHandler.export_to_csv(
    conversation_log,
    'output/conversation.csv',
    fields=['turn', 'from_user', 'original_message', 'translated_message']
)

# Load conversation data
data = DataHandler.load_conversation_data('results.json')

# Aggregate multiple evaluation results
aggregated = DataHandler.aggregate_results([
    'eval1.json',
    'eval2.json',
    'eval3.json'
])
```

## Configuration

### Translation Brief

The translation brief provides guidelines to the interpreter agent. Create a file with instructions like:

```text
You are an expert translator. Translate accurately while:
1. Preserving the original meaning
2. Maintaining tone and style
3. Adapting idioms appropriately
4. Being culturally sensitive
```

Load it with:
```python
brief = DataHandler.load_translation_brief('config/translation_brief.txt')
```

### User Context

User contexts help LLM-powered users maintain consistent personas:

```python
context_en = DataHandler.load_user_context('config/user_context_en.txt')
user = User(name="Alice", language="English", is_llm=True, 
            llm_provider=provider, context=context_en)
```

## Evaluation Metrics

The framework tracks:
- Total conversation turns
- Average translation time
- Language pairs used
- Conversation history
- Translation quality (can be extended with BLEU, COMET, etc.)

## Contributing

Contributions are welcome! Areas for enhancement:
- Additional LLM providers
- Advanced evaluation metrics (BLEU, COMET, BERTScore)
- Async conversation support
- Web interface
- More example scenarios

## License

See LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{interpreter_agent_eval,
  title={Interpreter Agent Evaluation Framework},
  author={Faiz Ghifari},
  year={2026},
  url={https://github.com/faizghifari/interpreter-agent-eval}
}
```

## Contact

For questions or issues, please open an issue on GitHub.
