"""Local GPU chat LLM interpreters (HF transformers, causal LM).

Unlike NLLB/Seamless (seq2seq direct NMT, ``local_nmt.py``), these are
instruction-tuned chat models, so they go through the *normal* interpreter
path: ``operations.translate_record`` builds the (system, user) translation
brief and calls ``provider.generate(user_prompt, system_prompt=...)``. This
class is a drop-in for the cloud ``LLMProvider`` API but runs the weights
locally in fp16 on CUDA.

Primary target: ``CohereLabs/tiny-aya-global`` (cohere2, 3.35B) — broad
multilingual coverage incl. Bengali, fits ~7 GB fp16. Reusable for other small
HF chat checkpoints (e.g. SEA-LION) via ``model_name``.
"""

from typing import Optional


class HFChatProvider:
    """Local HF causal-LM chat model exposing the LLMProvider.generate API."""

    def __init__(
        self,
        model_name: str = "CohereLabs/tiny-aya-global",
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._tok = None
        self._model = None

    def label(self) -> str:
        return f"aya:{self.model_name.split('/')[-1]}"

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._model = (
            AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype)
            .to(self.device)
            .eval()
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self._load()
        import torch

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        # transformers 5.x returns a BatchEncoding (dict-like), not a bare
        # tensor, so request the dict explicitly and pass the attention mask.
        enc = self._tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)
        input_len = enc["input_ids"].shape[1]
        gen_kwargs = dict(max_new_tokens=self.max_new_tokens)
        if self.temperature and self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            out = self._model.generate(**enc, **gen_kwargs)
        # Decode only the newly generated continuation.
        new_tokens = out[0][input_len:]
        return self._tok.decode(new_tokens, skip_special_tokens=True).strip()


def build_local_llm(provider_type: str, model_name: Optional[str] = None):
    """provider_type 'aya' -> tiny-aya-global (override checkpoint via model_name)."""
    if provider_type == "aya":
        return HFChatProvider(model_name or "CohereLabs/tiny-aya-global")
    raise ValueError(f"Unknown local LLM provider '{provider_type}' (use 'aya')")
