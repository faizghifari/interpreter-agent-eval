"""Local GPU translation models (direct NMT, not LLM chat).

NLLB-200 and SeamlessM4T-v2 are seq2seq translators: they take raw source text
plus source/target language codes — no translation brief or chat prompt. They
therefore bypass the InterpreterAgent prompt path and expose a uniform
``translate(text, src_iso3, tgt_iso3) -> str`` used by
``operations.translate_record_local``.

Language codes in the data are ISO 639-3 (arb/ben/ind/kor/eng). NLLB needs
FLORES-200 codes (e.g. kor_Hang); SeamlessM4T-v2 uses ISO 639-3 directly.
Models load lazily in fp16 on CUDA (fits a 16 GB card for NLLB-3.3B and
seamless-m4t-v2-large).
"""

from typing import Optional

# ISO 639-3 -> FLORES-200 code (NLLB)
_FLORES = {
    "arb": "arb_Arab",
    "ben": "ben_Beng",
    "ind": "ind_Latn",
    "kor": "kor_Hang",
    "eng": "eng_Latn",
}
# ISO 639-3 codes SeamlessM4T-v2 accepts directly for these languages.
_SEAMLESS = {"arb": "arb", "ben": "ben", "ind": "ind", "kor": "kor", "eng": "eng"}


class NLLBTranslator:
    """facebook/nllb-200-* dense translator."""

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-3.3B",
        device: str = "cuda",
        max_new_tokens: int = 512,
        num_beams: int = 4,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self._tok = None
        self._model = None

    def label(self) -> str:
        return f"nllb:{self.model_name.split('/')[-1]}"

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._model = (
            AutoModelForSeq2SeqLM.from_pretrained(self.model_name, torch_dtype=dtype)
            .to(self.device)
            .eval()
        )

    def translate(self, text: str, src_iso3: str, tgt_iso3: str) -> str:
        if src_iso3 not in _FLORES or tgt_iso3 not in _FLORES:
            raise ValueError(f"NLLB: unsupported language pair {src_iso3}->{tgt_iso3}")
        self._load()
        import torch

        # transformers 5.x no longer prepends the source-language token via
        # tok.src_lang, so the model loses source conditioning and degenerates
        # (echoes the input). Prepend the FLORES src token id manually.
        enc = self._tok(text, return_tensors="pt", truncation=True, max_length=1024)
        src_id = self._tok.convert_tokens_to_ids(_FLORES[src_iso3])
        input_ids = torch.cat([torch.tensor([[src_id]]), enc.input_ids], dim=1).to(
            self.device
        )
        attn = torch.ones_like(input_ids)
        bos = self._tok.convert_tokens_to_ids(_FLORES[tgt_iso3])
        with torch.no_grad():
            gen = self._model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                forced_bos_token_id=bos,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
            )
        return self._tok.batch_decode(gen, skip_special_tokens=True)[0].strip()


class SeamlessTranslator:
    """facebook/seamless-m4t-v2-large text-to-text translator."""

    def __init__(
        self,
        model_name: str = "facebook/seamless-m4t-v2-large",
        device: str = "cuda",
        max_new_tokens: int = 512,
        num_beams: int = 4,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self._proc = None
        self._model = None

    def label(self) -> str:
        return f"seamless:{self.model_name.split('/')[-1]}"

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, SeamlessM4Tv2Model

        self._proc = AutoProcessor.from_pretrained(self.model_name)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._model = (
            SeamlessM4Tv2Model.from_pretrained(self.model_name, torch_dtype=dtype)
            .to(self.device)
            .eval()
        )

    def translate(self, text: str, src_iso3: str, tgt_iso3: str) -> str:
        if src_iso3 not in _SEAMLESS or tgt_iso3 not in _SEAMLESS:
            raise ValueError(f"Seamless: unsupported pair {src_iso3}->{tgt_iso3}")
        self._load()
        import torch

        inputs = self._proc(
            text=text, src_lang=_SEAMLESS[src_iso3], return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                tgt_lang=_SEAMLESS[tgt_iso3],
                generate_speech=False,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
            )
        # text-to-text: ids come back as out[0] (tensor) or .sequences
        seq = getattr(out, "sequences", None)
        if seq is None:
            seq = out[0] if isinstance(out, (list, tuple)) else out
        ids = seq[0].tolist() if hasattr(seq[0], "tolist") else list(seq[0])
        return self._proc.decode(ids, skip_special_tokens=True).strip()


def build_local_translator(kind: str, model_name: Optional[str] = None):
    """kind in {'nllb','seamless'}; model_name overrides the default checkpoint."""
    if kind == "nllb":
        return NLLBTranslator(model_name or "facebook/nllb-200-3.3B")
    if kind == "seamless":
        return SeamlessTranslator(model_name or "facebook/seamless-m4t-v2-large")
    raise ValueError(f"Unknown local translator '{kind}' (use 'nllb' or 'seamless')")
