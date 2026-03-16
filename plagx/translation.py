"""
Translation service using Helsinki-NLP MarianMT models.
Translates foreign-language sentences to English so users can see
what the matched reference text actually says.

Features:
- Models loaded lazily and cached in memory
- Translations cached to disk (plagx/data/translation_cache/) so they are
  never recomputed across runs.  Cache key = SHA256(lang + sentences).
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List

from transformers import MarianMTModel, MarianTokenizer

from plagx.config import LANGUAGES, BASE_DIR

LOG = logging.getLogger(__name__)

# Disk cache directory
_CACHE_DIR = BASE_DIR / "data" / "translation_cache"

# Memory cache: lang_code -> (tokenizer, model)
_model_cache: Dict[str, tuple] = {}


def _cache_key(sentences: List[str], lang_code: str) -> str:
    payload = lang_code + "\n" + "\n".join(sentences)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _load_disk_cache(key: str) -> List[str] | None:
    path = _CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def _save_disk_cache(key: str, translations: List[str]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps(translations, ensure_ascii=False), encoding="utf-8")


def _get_model(lang_code: str):
    """Load and cache MarianMT model + tokenizer for lang_code -> English."""
    if lang_code in _model_cache:
        return _model_cache[lang_code]

    if lang_code not in LANGUAGES:
        raise ValueError(f"Unsupported language: {lang_code}")

    model_name = LANGUAGES[lang_code]["marian_model"]
    LOG.info("Loading translation model: %s", model_name)

    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
    except Exception as exc:  # network error, missing model, etc.
        LOG.warning("Failed to load translation model %s: %s. "
                    "Falling back to semantic-only scoring for [%s].",
                    model_name, exc, lang_code)
        _model_cache[lang_code] = (None, None)  # don't retry on next call
        return None, None

    _model_cache[lang_code] = (tokenizer, model)
    return tokenizer, model


def translate_to_english(sentences: List[str], lang_code: str,
                         batch_size: int = 32) -> List[str]:
    """
    Translate a list of sentences from `lang_code` to English.

    Results are cached to disk so subsequent runs skip inference entirely.

    Args:
        sentences: Source language sentences.
        lang_code: Source language code (e.g. 'nl', 'ja', 'de').
        batch_size: Batch size for translation (smaller = less RAM).

    Returns:
        List of English translations, same length as input.
    """
    if not sentences:
        return []

    key = _cache_key(sentences, lang_code)
    cached = _load_disk_cache(key)
    if cached is not None and len(cached) == len(sentences):
        LOG.info("[%s] Translation cache hit (%d sentences)", lang_code, len(sentences))
        return cached

    tokenizer, model = _get_model(lang_code)
    if tokenizer is None or model is None:
        # Model unavailable — return originals so semantic score still works
        return list(sentences)
    translations: List[str] = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=512)
        generated = model.generate(**encoded)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        translations.extend(decoded)

    _save_disk_cache(key, translations)
    return translations


def translate_single(text: str, lang_code: str) -> str:
    """Translate a single sentence/text from lang_code to English."""
    result = translate_to_english([text], lang_code)
    return result[0] if result else ""
