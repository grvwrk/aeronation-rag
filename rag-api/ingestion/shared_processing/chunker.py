"""Chunker: cleaned RawDoc -> Chunk objects.

Implements the 5-step chunking pipeline:
1. Paragraph split: Splits document into paragraph blocks.
2. Sentence split if necessary: Splits oversized paragraphs into sentences.
3. Pack into budget: Packs units into chunks adhering to `chunk_size`.
4. Add natural overlap: Starts subsequent chunks with trailing units from previous chunks up to `chunk_overlap`.
5. Chunk objects: Generates deterministic Chunk instances with inherited metadata and IDs.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure project root (rag-api) is on sys.path so top-level modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from errors import ChunkerError
from models import Chunk, ChunkingConfig, RawDoc, StageCounts

logger = logging.getLogger(__name__)

# Regex for sentence splitting: splits on punctuation (.!? or ellipses) followed by whitespace and capital/quote/number.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-Z0-9\"“'(\[])")

# Regex for paragraph splitting: two or more newlines.
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")


@dataclass
class _TextUnit:
    """An atomic text unit with its preceding separator for chunk assembly."""

    text: str
    is_paragraph_start: bool = False


def split_paragraphs(text: str) -> list[str]:
    """Split text into paragraph strings based on double newlines."""
    if not text:
        return []
    return [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text.strip()) if p.strip()]


def split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into individual sentence strings."""
    if not paragraph:
        return []
    # Split using sentence boundary regex while stripping residual whitespace
    raw_sentences = _SENTENCE_SPLIT_RE.split(paragraph.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences if sentences else [paragraph.strip()]


def _split_oversized_string(text: str, max_size: int) -> list[str]:
    """Split an oversized sentence or unbroken string into chunks <= max_size."""
    words = text.split(" ")
    pieces: list[str] = []
    current_piece: list[str] = []
    current_len = 0

    for word in words:
        if not word:
            continue
        # If single word is itself larger than max_size, hard-slice it
        if len(word) > max_size:
            if current_piece:
                pieces.append(" ".join(current_piece))
                current_piece = []
                current_len = 0
            for i in range(0, len(word), max_size):
                pieces.append(word[i : i + max_size])
            continue

        added_len = len(word) if not current_piece else current_len + 1 + len(word)
        if added_len <= max_size:
            current_piece.append(word)
            current_len = added_len
        else:
            if current_piece:
                pieces.append(" ".join(current_piece))
            current_piece = [word]
            current_len = len(word)

    if current_piece:
        pieces.append(" ".join(current_piece))

    return pieces


def _decompose_to_units(text: str, max_size: int) -> list[_TextUnit]:
    """Decompose document text into atomic _TextUnits (paragraphs, sentences, or word blocks)."""
    paragraphs = split_paragraphs(text)
    units: list[_TextUnit] = []

    for para in paragraphs:
        if len(para) <= max_size:
            units.append(_TextUnit(text=para, is_paragraph_start=True))
        else:
            # Oversized paragraph: split into sentences
            sentences = split_sentences(para)
            first_sentence = True
            for sentence in sentences:
                if len(sentence) <= max_size:
                    units.append(_TextUnit(text=sentence, is_paragraph_start=first_sentence))
                    first_sentence = False
                else:
                    # Oversized sentence: split into word blocks
                    sub_pieces = _split_oversized_string(sentence, max_size)
                    for piece in sub_pieces:
                        units.append(_TextUnit(text=piece, is_paragraph_start=first_sentence))
                        first_sentence = False

    return units


def _join_units(units: list[_TextUnit]) -> str:
    """Join a list of _TextUnits respecting paragraph and sentence boundaries."""
    if not units:
        return ""
    result: list[str] = []
    for i, unit in enumerate(units):
        if i == 0:
            result.append(unit.text)
        else:
            sep = "\n\n" if unit.is_paragraph_start else " "
            result.append(sep + unit.text)
    return "".join(result)


def pack_chunks_with_overlap(text: str, config: ChunkingConfig) -> list[str]:
    """Pack text into chunks respecting chunk_size budget and natural overlap."""
    if not text or not text.strip():
        return []

    chunk_size = config.chunk_size
    chunk_overlap = config.chunk_overlap

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")

    units = _decompose_to_units(text, chunk_size)
    if not units:
        return []

    chunks: list[str] = []
    start_idx = 0
    n_units = len(units)

    while start_idx < n_units:
        current_units: list[_TextUnit] = []
        current_len = 0
        end_idx = start_idx

        # Pack units greedily into chunk_size budget
        while end_idx < n_units:
            unit = units[end_idx]
            sep_len = 0
            if current_units:
                sep_len = 2 if unit.is_paragraph_start else 1

            candidate_len = current_len + sep_len + len(unit.text)
            if candidate_len <= chunk_size or not current_units:
                current_units.append(unit)
                current_len = candidate_len
                end_idx += 1
            else:
                break

        if current_units:
            chunk_str = _join_units(current_units).strip()
            if chunk_str:
                chunks.append(chunk_str)

        # If all units have been consumed, finish
        if end_idx >= n_units:
            break

        # Calculate overlap for the next chunk
        if chunk_overlap > 0 and end_idx > start_idx:
            overlap_units: list[_TextUnit] = []
            overlap_len = 0
            # Look backwards from end_idx - 1 to find trailing units fitting within chunk_overlap
            k = end_idx - 1
            while k >= start_idx:
                unit = units[k]
                sep_len = 0 if not overlap_units else (2 if overlap_units[0].is_paragraph_start else 1)
                candidate_overlap = overlap_len + sep_len + len(unit.text)
                if candidate_overlap <= chunk_overlap:
                    overlap_units.insert(0, unit)
                    overlap_len = candidate_overlap
                    k -= 1
                else:
                    break

            if overlap_units and len(overlap_units) < (end_idx - start_idx):
                start_idx = end_idx - len(overlap_units)
            else:
                # Ensure strictly monotonic forward progress
                start_idx = max(start_idx + 1, end_idx)
        else:
            start_idx = end_idx

    return chunks


import hashlib


def generate_chunk_id(text: str, prefix: str = "chunk") -> str:
    """Generate a deterministic, content-based chunk ID derived purely from the text content SHA-256 hash.

    Identical content across different files or sources produces the exact same hash and chunk_id,
    preventing duplicate chunk embeddings from receiving different IDs.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected text to be str, got {type(text).__name__}")
    digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}" if prefix else digest


def chunk_document(doc: RawDoc, config: ChunkingConfig | None = None) -> list[Chunk]:
    """Transform a single cleaned RawDoc into a list of Chunk objects."""
    if not isinstance(doc, RawDoc):
        raise ChunkerError(
            source=getattr(doc, "source_id", "unknown"),
            reason=f"Expected RawDoc instance, got {type(doc).__name__}",
        )

    if not isinstance(doc.text, str):
        page_num = doc.metadata.get("page_num") if isinstance(doc.metadata, dict) else None
        raise ChunkerError(
            source=doc.source_id,
            reason=f"RawDoc text must be a string, got {type(doc.text).__name__}",
            page=page_num,
        )

    config = config if config is not None else ChunkingConfig.from_env()
    page_num = doc.metadata.get("page_num") if isinstance(doc.metadata, dict) else None

    try:
        chunk_texts = pack_chunks_with_overlap(doc.text, config)
    except Exception as exc:
        raise ChunkerError(
            source=doc.source_id,
            reason=f"Error packing chunks: {exc}",
            page=page_num,
        ) from exc

    total_chunks = len(chunk_texts)
    chunks: list[Chunk] = []

    for idx, text in enumerate(chunk_texts, start=1):
        chunk_id = generate_chunk_id(text)
        metadata = dict(doc.metadata) if doc.metadata is not None else {}
        metadata.update({
            "chunk_id": chunk_id,
            "chunk_index": idx,
            "total_chunks": total_chunks,
            "char_count": len(text),
            "content_hash": hashlib.sha256(text.strip().encode("utf-8")).hexdigest(),
        })

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=text,
                source_id=doc.source_id,
                page_num=page_num,
                metadata=metadata,
            )
        )

    return chunks



def chunk_documents(
    docs: Iterable[RawDoc],
    config: ChunkingConfig | None = None,
    *,
    counts: StageCounts | None = None,
    strict: bool = False,
) -> list[Chunk]:
    """Transform an iterable of RawDoc instances into Chunk objects.

    - Updates StageCounts (chunks_created) if provided.
    - In strict=False mode (default), unexpected errors on individual documents are logged
      and skipped. In strict=True mode, ChunkerError is raised immediately.
    """
    counts = counts if counts is not None else StageCounts()
    config = config if config is not None else ChunkingConfig()
    all_chunks: list[Chunk] = []

    for doc in docs:
        try:
            doc_chunks = chunk_document(doc, config)
            all_chunks.extend(doc_chunks)
        except ChunkerError as exc:
            if strict:
                raise
            logger.error(str(exc.failure))
        except Exception as exc:
            source = getattr(doc, "source_id", "unknown")
            page = None
            if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                page = doc.metadata.get("page_num")
            err = ChunkerError(source=source, reason=f"Unexpected error during chunking: {exc}", page=page)
            if strict:
                raise err from exc
            logger.error(str(err.failure))

    counts.chunks_created += len(all_chunks)
    logger.info("created %d chunks from documents", len(all_chunks))
    return all_chunks
