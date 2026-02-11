# Interpreter Agent Evaluation Framework

A comprehensive framework for evaluating LLM-powered interpreter agents that facilitate communication between users speaking different languages.

## Features

- **Multiple Providers**: Google AI Studio, OpenAI, OpenRouter, vLLM.
- **Flexible Interactions**: Supports Human-to-Human, Human-to-AI, and AI-to-AI simulations.
- **Data Analysis**: Track translation quality and export conversation logs.
- **Standardized**: Uses ISO 639-3 language codes (e.g., `eng`, `spa`, `fra`).

## Installation

This project is managed with [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/faizghifari/interpreter-agent-eval.git
cd interpreter-agent-eval

# Install dependencies
uv sync
```

Alternatively, you can install with pip:
```bash
pip install -e .
```

## Quick Start

1. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Add your API keys (OPENAI_API_KEY, GOOGLE_API_KEY, etc.)
   ```

2. **Basic Usage**:

   ```python
   import os
   from interpreter_agent_eval import User, InterpreterAgent, EvaluationFramework
   from interpreter_agent_eval.providers import OpenAIProvider

   # 1. Setup Provider
   provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4o")

   # 2. Setup Users (using ISO 639-3 codes)
   user1 = User("Alice", "eng", is_llm=False)
   user2 = User("Carlos", "spa", is_llm=False)

   # 3. Setup Interpreter
   interpreter = InterpreterAgent(
       llm_provider=provider,
       translation_brief="Translate naturally and maintain tone.",
       source_language="eng",
       target_language="spa"
   )

   # 4. Run Conversation
   framework = EvaluationFramework(user1, user2, interpreter)
   messages = ["Hello!", "Hola, ¿cómo estás?", "I am good, thanks!"]
   conversation = framework.run_conversation(messages=messages)

   # 5. Export Results
   framework.export_results("results.json")
   ```

## Project Structure

```
├── config/         # Configuration templates & contexts
├── examples/       # Usage examples (basic, advanced, data)
├── src/            # Source code
├── tests/          # Unit tests
└── pyproject.toml  # Dependencies & metadata
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{interpreter_agent_eval,
  title={Interpreter Agent Evaluation Framework},
  author={Faiz Ghifari Haznitrama},
  year={2026},
  url={https://github.com/faizghifari/interpreter-agent-eval}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
