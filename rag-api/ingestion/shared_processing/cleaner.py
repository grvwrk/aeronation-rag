"""Cleaner: RawDoc -> cleaned RawDoc.

Performs practical, deterministic text cleanup on loaded RawDocs before chunking.
Normalizes line endings, cleans control characters, repairs line-break hyphenation,
collapses excessive whitespace, and preserves paragraph boundaries, technical Unicode,
equations, punctuation, and document metadata.
"""

from __future__ import annotations

import logging
import re
import sys
import unicodedata
from collections.abc import Iterable
from pathlib import Path

# Ensure project root (rag-api) is on sys.path so top-level modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from errors import CleanerError
from models import RawDoc, StageCounts

logger = logging.getLogger(__name__)

# Conservative hyphenation pattern for line-broken words (e.g. "sen-\ntence" -> "sentence").
_HYPHENATED_LINEBREAK_RE = re.compile(r"(?<=[a-zA-Z])-[ \t]*\n[ \t]*(?=[a-z])")

# Consecutive horizontal whitespace (spaces, tabs).
_HORIZONTAL_WHITESPACE_RE = re.compile(r"[^\S\n]+")

# Paragraph separator: two or more newlines with optional intervening whitespace.
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")


def normalize_line_endings(text: str) -> str:
    """Normalize Windows (\r\n) and legacy Mac (\r) line endings to Unix (\n)."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def remove_control_characters(text: str) -> str:
    """Remove non-printable control characters while preserving newlines, tabs,

    mathematical symbols, Greek letters, degree symbols, superscripts/subscripts,
    and Unicode punctuation.
    """
    return "".join(
        ch for ch in text if ch in ("\n", "\t") or unicodedata.category(ch) != "Cc"
    )


def normalize_hyphenated_line_breaks(text: str) -> str:
    """Conservatively remove hyphenation caused by PDF line-wrapping.

    Transforms 'sen-\\ntence' into 'sentence' without altering real hyphenated
    words such as 'state-of-the-art\\nmodel'.
    """
    return _HYPHENATED_LINEBREAK_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Normalize excessive whitespace while preserving meaningful paragraph boundaries.

    - Joins lines within the same paragraph with a single space.
    - Collapses multiple horizontal spaces and tabs into a single space.
    - Normalizes repeated blank lines so paragraphs are separated by exactly '\\n\\n'.
    """
    if not text:
        return ""

    paragraphs = _PARAGRAPH_SPLIT_RE.split(text.strip())
    cleaned_paragraphs: list[str] = []

    for para in paragraphs:
        lines = para.split("\n")
        cleaned_lines: list[str] = []
        for line in lines:
            line_clean = _HORIZONTAL_WHITESPACE_RE.sub(" ", line).strip()
            if line_clean:
                cleaned_lines.append(line_clean)

        if cleaned_lines:
            cleaned_paragraphs.append(" ".join(cleaned_lines))

    return "\n\n".join(cleaned_paragraphs)


def clean_text(text: str) -> str:
    """Execute the full text cleaning pipeline on a raw text string.

    1. Normalize line endings (\\r\\n, \\r -> \\n).
    2. Remove unwanted control characters (preserving \\n, \\t, Unicode).
    3. Repair line-break hyphenation artifacts.
    4. Normalize horizontal whitespace and preserve paragraph boundaries.
    5. Strip leading/trailing document whitespace.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected text to be str, got {type(text).__name__}")

    text = normalize_line_endings(text)
    text = remove_control_characters(text)
    text = normalize_hyphenated_line_breaks(text)
    text = normalize_whitespace(text)
    return text.strip()


def clean_document(doc: RawDoc) -> RawDoc | None:
    """Clean a single RawDoc.

    Returns a new RawDoc with cleaned text and preserved source_id and metadata.
    If the document contains no meaningful text after cleaning, returns None.
    """
    if not isinstance(doc, RawDoc):
        raise CleanerError(
            source=getattr(doc, "source_id", "unknown"),
            reason=f"Expected RawDoc instance, got {type(doc).__name__}",
        )

    if not isinstance(doc.text, str):
        page_num = doc.metadata.get("page_num") if isinstance(doc.metadata, dict) else None
        raise CleanerError(
            source=doc.source_id,
            reason=f"RawDoc text must be a string, got {type(doc.text).__name__}",
            page=page_num,
        )

    cleaned_text = clean_text(doc.text)
    if not cleaned_text:
        return None

    return RawDoc(
        text=cleaned_text,
        source_id=doc.source_id,
        metadata=dict(doc.metadata) if doc.metadata is not None else {},
    )


def clean_documents(
    docs: Iterable[RawDoc],
    *,
    counts: StageCounts | None = None,
    strict: bool = False,
) -> list[RawDoc]:
    """Clean a sequence of RawDoc instances.

    - Strips artifacts and normalizes whitespace while preserving paragraphs and metadata.
    - Discards empty / whitespace-only documents (recorded in counts.docs_discarded).
    - Updates StageCounts (docs_cleaned and docs_discarded) if provided.
    - In strict=False mode (default), unexpected errors on individual documents are logged
      and discarded. In strict=True mode, CleanerError is raised immediately.
    """
    counts = counts if counts is not None else StageCounts()
    cleaned_docs: list[RawDoc] = []

    for doc in docs:
        try:
            cleaned = clean_document(doc)
            if cleaned is None:
                counts.docs_discarded += 1
            else:
                cleaned_docs.append(cleaned)
                counts.docs_cleaned += 1
        except CleanerError as exc:
            if strict:
                raise
            logger.error(str(exc.failure))
            counts.docs_discarded += 1
        except Exception as exc:
            source = getattr(doc, "source_id", "unknown")
            page = None
            if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                page = doc.metadata.get("page_num")
            err = CleanerError(source=source, reason=f"Unexpected error during cleaning: {exc}", page=page)
            if strict:
                raise err from exc
            logger.error(str(err.failure))
            counts.docs_discarded += 1

    logger.info("cleaned %d documents, discarded %d empty documents", len(cleaned_docs), counts.docs_discarded)
    return cleaned_docs
