# Interpreter Agent Evaluation Framework

A comprehensive framework for evaluating LLM-powered interpreter agents that facilitate communication between users speaking different languages.

## Features

- **Multiple Providers**: Google AI Studio (Gemini 3 Pro/Flash), OpenAI (GPT-4o), Friendli (EXAONE), OpenRouter, vLLM.
- **AI-as-Judge Evaluation**: Integrated evaluation framework with structured verification checklists and "Yes/No" metrics.
- **Language Verification**: Built-in support for GlotLID to ensure users maintain monolingual behavior and avoid code-switching.
- **Flexible Interactions**: Supports Human-to-Human, Human-to-AI, and AI-to-AI simulations.
- **Standardized**: Uses ISO 639-3 three-letter language codes (e.g., `eng`, `spa`, `ind`, `kor`, `arb`).

## Installation

This project is managed with [uv](https://github.com/astral-sh/uv).

```bash
# Install dependencies
uv sync

# Run evaluation scripts
uv run python scripts/run_custom_eval.py
```

## Quick Start

### 1. Configure Environment
```bash
cp .env.example .env
# Add your API keys (OPENAI_API_KEY, GEMINI_API_KEY, FRIENDLI_TOKEN, etc.)
```

### 2. Run Automated Evaluation
The framework is designed for large-scale evaluation using predefined scenarios in JSONL format.

```bash
uv run python scripts/run_custom_eval.py --data data/enriched/id_kr.jsonl --num_samples 5
```

### 3. Basic Library Usage (Programmatic)

```python
import os
from interpreter_agent_eval import User, InterpreterAgent, EvaluationFramework
from interpreter_agent_eval.providers import GoogleAIProvider

# 1. Setup Provider
provider = GoogleAIProvider(api_key=os.getenv("GEMINI_API_KEY"), model_name="gemini-3-flash-preview")

# 2. Setup Users (using ISO 639-3 codes)
user1 = User("Alice", "eng", is_llm=False)
user2 = User("Carlos", "ind", is_llm=True, llm_provider=provider)

# 3. Setup Interpreter
interpreter = InterpreterAgent(
    llm_provider=provider,
    source_language="eng",
    target_language="ind"
)

# 4. Run Conversation
framework = EvaluationFramework(user1, user2, interpreter)
conversation = framework.run_conversation(messages=["I'd like to book a room."])

# 5. Evaluate with a Judge
judge_provider = GoogleAIProvider(api_key=os.getenv("GEMINI_API_KEY"), model_name="gemini-3-pro-preview")
evaluation = framework.evaluate_with_judge(
    judge_llm_provider=judge_provider,
    verification_prompt="1. Did the interpreter mention booking a room?\n2. Is the translation in Indonesian?"
)
print(f"Completion Rate: {evaluation.get_completion_rate()}")
```

## Project Structure

```
├── data/           # Evaluation datasets (CSV/JSONL)
├── scripts/        # Execution scripts for simulation and analysis
├── src/            # Core framework source code
│   ├── models/     # Pydantic models for structured output
│   ├── providers/  # LLM provider implementations
│   └── utils/      # Language detection and data handling
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
