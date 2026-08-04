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

from typing import List, Optional, Tuple


class HFChatProvider:
    """Local HF causal-LM chat model exposing the LLMProvider.generate API."""

    def __init__(
        self,
        model_name: str = "CohereLabs/tiny-aya-global",
        device: str = "cuda",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        batch_size: int = 8,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size
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
        load_kwargs = {"dtype": dtype, "low_cpu_mem_usage": True}
        if self.device.startswith("cuda"):
            # Load shards directly onto the GPU. Loading the full fp16 model
            # into host RAM and then copying it with .to("cuda") briefly keeps
            # two large copies alive and can trigger the WSL OOM killer.
            load_kwargs["device_map"] = self.device
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **load_kwargs
        ).eval()

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        return self.generate_batch([(prompt, system_prompt)])[0]

    def generate_batch(
        self, requests: List[Tuple[str, Optional[str]]]
    ) -> List[str]:
        """Generate several independent chat completions in GPU batches."""
        self._load()
        import torch

        rendered = []
        for prompt, system_prompt in requests:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            rendered.append(
                self._tok.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            )

        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        old_padding_side = self._tok.padding_side
        self._tok.padding_side = "left"
        results: List[str] = []
        try:
            for start in range(0, len(rendered), self.batch_size):
                batch = rendered[start : start + self.batch_size]
                enc = self._tok(
                    batch, padding=True, return_tensors="pt", add_special_tokens=False
                ).to(self.device)
                input_len = enc["input_ids"].shape[1]
                gen_kwargs = dict(
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self._tok.pad_token_id,
                )
                if self.temperature and self.temperature > 0:
                    gen_kwargs.update(do_sample=True, temperature=self.temperature)
                else:
                    gen_kwargs.update(do_sample=False)
                with torch.no_grad():
                    out = self._model.generate(**enc, **gen_kwargs)
                for sequence in out:
                    results.append(
                        self._tok.decode(
                            sequence[input_len:], skip_special_tokens=True
                        ).strip()
                    )
        finally:
            self._tok.padding_side = old_padding_side
        return results


def build_local_llm(provider_type: str, model_name: Optional[str] = None):
    """provider_type 'aya' -> tiny-aya-global (override checkpoint via model_name)."""
    if provider_type == "aya":
        return HFChatProvider(model_name or "CohereLabs/tiny-aya-global")
    raise ValueError(f"Unknown local LLM provider '{provider_type}' (use 'aya')")
