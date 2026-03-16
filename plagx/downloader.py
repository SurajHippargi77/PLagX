"""
Download real academic reference documents from Wikipedia in each target language.
Stores them as .txt files under plagx/data/reference_papers/<lang_code>/.

Each language gets articles on 3 academic topics in that language's native titles,
so Wikipedia finds the correct articles for each language edition.
"""

import logging
from pathlib import Path
from typing import Dict, List

import wikipediaapi

from plagx.config import LANGUAGES, REFERENCE_TOPICS_BY_LANG, DATA_DIR

LOG = logging.getLogger(__name__)


def _safe_filename(title: str) -> str:
    """Convert a Wikipedia title into a safe filename."""
    # Replace spaces and slashes with underscores, keep alphanumeric + underscore
    import re
    name = re.sub(r'[^\w\s-]', '', title)
    name = re.sub(r'[\s]+', '_', name).strip('_')
    return name[:80]  # truncate long titles


def download_documents(languages: Dict | None = None,
                       force: bool = False) -> Dict[str, List[Path]]:
    """
    Download Wikipedia articles for each language using native-language article titles.

    Args:
        languages: Override LANGUAGES config. Default: all 5 languages.
        force: Re-download even if files exist.

    Returns:
        Dict mapping lang_code -> list of saved file paths.
    """
    languages = languages or LANGUAGES
    result: Dict[str, List[Path]] = {}

    for lang_code, lang_info in languages.items():
        wiki_code = lang_info["wiki_code"]
        lang_name = lang_info["name"]
        lang_dir = DATA_DIR / lang_code
        lang_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []

        # Use native-language titles for this language
        topics = REFERENCE_TOPICS_BY_LANG.get(lang_code, [])
        if not topics:
            LOG.warning("[%s] No topics configured, skipping.", lang_code)
            result[lang_code] = []
            continue

        wiki = wikipediaapi.Wikipedia(
            user_agent="PLagX/1.0 (academic plagiarism research)",
            language=wiki_code,
        )

        for topic in topics:
            safe_name = _safe_filename(topic)
            out_path = lang_dir / f"{safe_name}.txt"

            if out_path.exists() and not force:
                LOG.info("[%s] Already exists: %s", lang_code, out_path.name)
                saved.append(out_path)
                continue

            page = wiki.page(topic)
            if not page.exists():
                LOG.warning("[%s] Topic '%s' not found on %s Wikipedia", lang_code, topic, lang_name)
                continue

            content = page.text
            if not content or len(content) < 100:
                LOG.warning("[%s] Topic '%s' has too little content (%d chars)", lang_code, topic, len(content or ""))
                continue

            out_path.write_text(content, encoding="utf-8")
            LOG.info("[%s] Downloaded '%s': %d chars", lang_code, topic, len(content))
            saved.append(out_path)

        result[lang_code] = saved

    return result


def list_downloaded() -> Dict[str, List[Path]]:
    """List all currently downloaded reference documents by language."""
    result: Dict[str, List[Path]] = {}
    if not DATA_DIR.exists():
        return result
    for lang_code in LANGUAGES:
        lang_dir = DATA_DIR / lang_code
        if lang_dir.is_dir():
            files = sorted(lang_dir.glob("*.txt"))
            if files:
                result[lang_code] = files
    return result
