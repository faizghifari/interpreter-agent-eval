"""Free web-endpoint translators (Google Translate, Papago).

These hit the public consumer web endpoints rather than the paid official
APIs, so they need no key. Both expose the same
``translate(text, src_iso3, tgt_iso3) -> str`` contract as ``local_nmt`` and
run through ``operations.translate_record_local`` (same mojibake guard +
pending-retry semantics). Transient throttling (HTTP 429 / blocks) raises and
routes the record to ``<output>.pending.jsonl`` for a later re-run.

Language coverage note: Papago does NOT support Bengali (and Arabic only
unofficially), so among arb/ben/ind/kor it reliably covers ind<->kor only;
other pairs raise ``ValueError`` (-> permanent pending). Feed Papago an
ind/kor-filtered input to avoid retrying unsupported records.
"""

# ISO 639-3 -> 2-letter codes the web endpoints use.
_GOOGLE = {"arb": "ar", "ben": "bn", "ind": "id", "kor": "ko", "eng": "en"}
_PAPAGO = {"ind": "id", "kor": "ko"}  # reliable web coverage for our langs

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class GoogleWebTranslator:
    """Free, key-less Google Translate via the translate_a/single endpoint."""

    _URL = "https://translate.googleapis.com/translate_a/single"

    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout
        self._client = None

    def label(self) -> str:
        return "google:web-free"

    def _get_client(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(
                timeout=self.timeout, headers={"User-Agent": _UA}
            )
        return self._client

    def translate(self, text: str, src_iso3: str, tgt_iso3: str) -> str:
        if src_iso3 not in _GOOGLE or tgt_iso3 not in _GOOGLE:
            raise ValueError(f"Google: unsupported pair {src_iso3}->{tgt_iso3}")
        if not text or not text.strip():
            return ""
        params = {
            "client": "gtx",
            "sl": _GOOGLE[src_iso3],
            "tl": _GOOGLE[tgt_iso3],
            "dt": "t",
            "q": text,
        }
        resp = self._get_client().get(self._URL, params=params)
        resp.raise_for_status()  # 429/5xx -> raise -> pending -> retry
        data = resp.json()
        # data[0] is a list of [translated_segment, source_segment, ...].
        segments = data[0] or []
        return "".join(seg[0] for seg in segments if seg and seg[0]).strip()


class PapagoWebTranslator:
    """Free, key-less Papago via the PentaGo reverse-engineered web wrapper."""

    def __init__(self):
        self._engines = {}  # (src2,tgt2) -> Pentago instance
        self._lang = None

    def label(self) -> str:
        return "papago:web-free"

    def _const(self, code2: str):
        # pentago.lang exposes per-language constants; map our 2-letter codes.
        if self._lang is None:
            import pentago.lang as lang

            self._lang = lang
        names = {"ko": "KOREAN", "id": "INDONESIAN"}
        return getattr(self._lang, names[code2])

    def _engine(self, src2: str, tgt2: str):
        key = (src2, tgt2)
        if key not in self._engines:
            from pentago import Pentago

            self._engines[key] = Pentago(self._const(src2), self._const(tgt2))
        return self._engines[key]

    def translate(self, text: str, src_iso3: str, tgt_iso3: str) -> str:
        if src_iso3 not in _PAPAGO or tgt_iso3 not in _PAPAGO:
            raise ValueError(
                f"Papago: unsupported pair {src_iso3}->{tgt_iso3} "
                f"(web coverage limited to ind<->kor)"
            )
        if not text or not text.strip():
            return ""
        eng = self._engine(_PAPAGO[src_iso3], _PAPAGO[tgt_iso3])
        result = eng.translate_sync(text)
        # PentaGo returns either a plain string or an object with .translated.
        return (getattr(result, "translated", None) or str(result)).strip()


def build_web_translator(kind: str):
    """kind in {'google','papago'}."""
    if kind == "google":
        return GoogleWebTranslator()
    if kind == "papago":
        return PapagoWebTranslator()
    raise ValueError(f"Unknown web translator '{kind}' (use 'google' or 'papago')")
