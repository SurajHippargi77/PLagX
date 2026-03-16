"""
PLagX Configuration.
Central settings for the cross-lingual plagiarism detection engine.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "reference_papers"

# ── Supported languages ─────────────────────────────────────────────────────
# Each entry: language_code -> (display_name, wikipedia_lang_code, marian_model_name)
# MarianMT model naming: Helsinki-NLP/opus-mt-{src}-{tgt}
LANGUAGES = {
    "nl": {
        "name": "Dutch",
        "wiki_code": "nl",
        "marian_model": "Helsinki-NLP/opus-mt-nl-en",
    },
    "ja": {
        "name": "Japanese",
        "wiki_code": "ja",
        "marian_model": "Helsinki-NLP/opus-mt-ja-en",
    },
    "de": {
        "name": "German",
        "wiki_code": "de",
        "marian_model": "Helsinki-NLP/opus-mt-de-en",
    },
    "es": {
        "name": "Spanish",
        "wiki_code": "es",
        "marian_model": "Helsinki-NLP/opus-mt-es-en",
    },
    "pt": {
        "name": "Portuguese",
        "wiki_code": "pt",
        "marian_model": "Helsinki-NLP/opus-mt-ROMANCE-en",  # covers pt, es, fr, it
    },
}

# ── NLP Models ───────────────────────────────────────────────────────────────
# Multilingual sentence embedding model (supports 50+ languages)
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

# ── Similarity Thresholds ────────────────────────────────────────────────────
# Semantic embedding similarity threshold (0-1). Sentences above this are flagged.
SEMANTIC_THRESHOLD = float(os.environ.get("PLAGX_SEMANTIC_THRESHOLD", "0.72"))

# TF-IDF cosine similarity threshold for translated text comparison
TFIDF_THRESHOLD = float(os.environ.get("PLAGX_TFIDF_THRESHOLD", "0.55"))

# Combined score weights: final_score = w_semantic * semantic + w_tfidf * tfidf
W_SEMANTIC = 0.7
W_TFIDF = 0.3

# Final combined threshold for flagging a sentence as plagiarized
COMBINED_THRESHOLD = float(os.environ.get("PLAGX_COMBINED_THRESHOLD", "0.62"))

# ── Wikipedia topics for reference documents ─────────────────────────────────
# Native-language article titles for Wikipedia per language.
# Each language gets the same 3 academic topics.
# Titles must match the article title in THAT language's Wikipedia.
REFERENCE_TOPICS_BY_LANG: dict = {
    "nl": [
        "Kunstmatige intelligentie",   # Artificial intelligence
        "Klimaatverandering",           # Climate change
        "Hernieuwbare energie",         # Renewable energy
    ],
    "ja": [
        "人工知能",                       # Artificial intelligence
        "気候変動",                       # Climate change
        "再生可能エネルギー",              # Renewable energy
    ],
    "de": [
        "Künstliche Intelligenz",       # Artificial intelligence
        "Klimawandel",                  # Climate change
        "Erneuerbare Energie",          # Renewable energy
    ],
    "es": [
        "Inteligencia artificial",      # Artificial intelligence
        "Cambio climático",             # Climate change
        "Energía renovable",            # Renewable energy
    ],
    "pt": [
        "Inteligência artificial",      # Artificial intelligence
        "Mudanças climáticas",          # Climate change
        "Energia renovável",            # Renewable energy
    ],
}

# Maximum sentences per reference document
MAX_SENTENCES_PER_DOC = 200
