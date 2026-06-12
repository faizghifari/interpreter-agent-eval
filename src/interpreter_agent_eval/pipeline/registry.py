"""Provider construction for pipeline stages.

Centralizes the model wiring that used to live inline in ``scripts/run_eval.py``:
the cloud interpreter/judge providers, and the per-language local user-simulation
providers (LM Studio endpoints).
"""

import os
from typing import Callable, Tuple

from interpreter_agent_eval.providers import (
    GoogleAIProvider,
    OpenAIProvider,
    OpenRouterProvider,
)

# ---------------------------------------------------------------------------
# Defaults (kept in sync with the historical run_eval defaults)
# ---------------------------------------------------------------------------
DEFAULT_INTERPRETER_PROVIDER = "gemini"
DEFAULT_INTERPRETER_MODEL = "gemini-3.1-flash-lite-preview"

DEFAULT_JUDGE_PROVIDER = "gemini"
DEFAULT_JUDGE_MODEL = "gemini-3.1-pro-preview"
DEFAULT_JUDGE_THINKING_LEVEL = "high"

LANG_FULL = {
    "arb": "Arabic",
    "ind": "Indonesian",
    "kor": "Korean",
    "ben": "Bengali",
}


# ---------------------------------------------------------------------------
# Cloud providers (interpreter / judge) — batchable in a later phase
# ---------------------------------------------------------------------------
def build_interpreter_provider(
    provider_type: str,
    model_name: str,
    thinking_level: str = "minimal",
):
    """Build an interpreter LLM provider from CLI-supplied type and model name."""
    if provider_type == "gemini":
        if thinking_level == "none":
            return GoogleAIProvider(
                model_name=model_name, http_options={"timeout": 120000}
            )
        thinking_config = {
            "thinking_config": {
                "include_thoughts": True,
                "thinking_level": thinking_level,
            }
        }
        return GoogleAIProvider(
            model_name=model_name, http_options={"timeout": 120000}, **thinking_config
        )
    elif provider_type == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment")
        return OpenRouterProvider(
            api_key=api_key, model_name=model_name, app_name="mt-eval"
        )
    elif provider_type == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        return OpenAIProvider(api_key=api_key, model_name=model_name)
    raise ValueError(
        f"Unknown interpreter provider '{provider_type}'. Choose from: gemini, openrouter, openai"
    )


def build_judge_provider(
    provider_type: str,
    model_name: str,
    thinking_level: str = DEFAULT_JUDGE_THINKING_LEVEL,
):
    """Build a judge LLM provider from CLI-supplied type, model, and thinking level."""
    if provider_type == "gemini":
        thinking_config = {
            "thinking_config": {
                "include_thoughts": True,
                "thinking_level": thinking_level,
            }
        }
        return GoogleAIProvider(
            model_name=model_name, http_options={"timeout": 120000}, **thinking_config
        )
    elif provider_type == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment")
        return OpenRouterProvider(
            api_key=api_key, model_name=model_name, app_name="mt-eval"
        )
    elif provider_type == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        return OpenAIProvider(api_key=api_key, model_name=model_name)
    raise ValueError(
        f"Unknown judge provider '{provider_type}'. Choose from: gemini, openrouter, openai"
    )


# ---------------------------------------------------------------------------
# Local user-simulation providers (LM Studio) — not batchable
# ---------------------------------------------------------------------------
def _lm_studio_base_url() -> str:
    return os.getenv("LM_STUDIO_BASE_URL", "http://14.50.130.58:1234/v1")


def create_id_model_provider() -> OpenAIProvider:
    # SEA-LION v4 Qwen VL 8B — SEA languages only. max_tokens caps runaway loops.
    return OpenAIProvider(
        api_key="lm-studio",
        model_name=os.getenv("LM_STUDIO_ID_MODEL", "qwen-sea-lion-v4-8b-vl"),
        base_url=_lm_studio_base_url(),
        max_tokens=1024,
    )


def create_kr_model_provider() -> OpenAIProvider:
    # EXAONE-3.5-7.8B — Korean/English bilingual.
    return OpenAIProvider(
        api_key="lm-studio",
        model_name=os.getenv("LM_STUDIO_KR_MODEL", "exaone-3.5-7.8b-instruct"),
        base_url=_lm_studio_base_url(),
    )


def create_ar_model_provider() -> OpenAIProvider:
    # Command-R7B Arabic (swap in LM Studio before Arabic runs).
    return OpenAIProvider(
        api_key="lm-studio",
        model_name=os.getenv("LM_STUDIO_AR_MODEL", "c4ai-command-r7b-arabic-02-2025"),
        base_url=_lm_studio_base_url(),
    )


def create_bn_model_provider() -> OpenAIProvider:
    return OpenAIProvider(
        api_key="lm-studio",
        model_name=os.getenv("LM_STUDIO_BN_MODEL", "bengali-llm-placeholder"),
        base_url=_lm_studio_base_url(),
    )


_USER_SIM_BUILDERS = {
    "ind": create_id_model_provider,
    "kor": create_kr_model_provider,
    "arb": create_ar_model_provider,
    "ben": create_bn_model_provider,
}


def make_user_sim_factory() -> Callable[[str], Tuple[object, str, str]]:
    """Return a factory ``lang -> (provider, model_label, language_name)``.

    Providers are cached per language so the underlying HTTP client is reused
    across records within a run.
    """
    cache: dict = {}

    def factory(lang: str) -> Tuple[object, str, str]:
        if lang not in cache:
            builder = _USER_SIM_BUILDERS.get(lang)
            if builder is None:
                raise ValueError(
                    f"No user-simulation provider configured for language '{lang}'"
                )
            provider = builder()
            label = getattr(provider, "model_name", lang)
            cache[lang] = (provider, label, LANG_FULL.get(lang, lang))
        return cache[lang]

    return factory


def label_for(provider_type: str, model_name: str) -> str:
    """Human-readable ``provider:model`` label used in output records."""
    return f"{provider_type}:{model_name}"
