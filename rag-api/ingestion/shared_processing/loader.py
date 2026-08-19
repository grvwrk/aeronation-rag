"""Loader: raw source -> RawDoc per page.

Shared by both pipelines. The system pipeline points it at a directory (and,
now, optionally a list of URLs); the user pipeline hands it a single uploaded
file or URL. That is the entire difference, which is why this is a function
and not a class hierarchy.

Page-level granularity is deliberate and non-negotiable: `page_num` in the
metadata is what the citation template renders. Load whole documents and you
cannot get it back without re-parsing. Formats without a native page concept
(docx, webpages, single images) are treated as one logical page — see the
docstrings on `_load_docx` / `_load_webpage` / `_load_image` for why.

Every failure raised by this module is a LoaderError subclass carrying
source/page/reason context (see ../errors.py) — never a bare stdlib exception
like IndexError. In non-strict mode those failures are logged in the same
structured form and the source is skipped, so one bad file in a corpus of
forty doesn't kill a twenty-minute ingestion run.

Optional dependencies (only required for the formats you actually use):
    pdf            pip install pypdf
    pdf OCR        pip install pdf2image pytesseract   + poppler + tesseract (system binaries)
    png/jpg OCR    pip install Pillow pytesseract       + tesseract (system binary)
    docx           pip install python-docx
    webpages       pip install requests trafilatura     (falls back to beautifulsoup4 if trafilatura is absent)
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from urllib.parse import urlparse

# Ensure project root (rag-api) is on sys.path so top-level modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from errors import (
    EmptySourceError,
    LoaderError,
    ParseError,
    SourceError,
    SourceNotFoundError,
    SourcePermissionError,
    UnsupportedFormatError,
)
from models import RawDoc, StageCounts

logger = logging.getLogger(__name__)

# Formats the loader can actually parse.
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg", ".docx"}

# Formats we recognize but deliberately don't support, with a helpful reason
# instead of a generic "unsupported format" message.
_KNOWN_UNSUPPORTED = {
    ".doc": "legacy binary .doc is not supported — convert to .docx or .pdf first",
    ".ppt": "PowerPoint is not a supported source format",
    ".pptx": "PowerPoint is not a supported source format",
    ".html": "raw local .html files aren't handled — load the live page as a URL, "
    "or rename/convert to .txt if it's already plain text",
}

# Below this many non-whitespace characters, a PDF page is treated as
# "probably scanned" rather than "genuinely blank", and gets an OCR pass.
_OCR_MIN_CHARS = 20

_ocr_deps_warned = False


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _is_url(value: str) -> bool:
    return isinstance(value, str) and re.match(r"^https?://", value.strip(), re.IGNORECASE) is not None


def _validate_local_source(path: Path) -> None:
    """Exists? Readable? Non-empty? Raises a typed LoaderError if not.

    Format support is checked separately by the caller, since it doesn't
    require touching the filesystem.
    """
    if not path.exists():
        raise SourceNotFoundError(str(path), "file does not exist")
    if not path.is_file():
        raise SourceNotFoundError(str(path), "path exists but is not a file (is it a directory?)")
    if not os.access(path, os.R_OK):
        raise SourcePermissionError(str(path), "file exists but is not readable (permission denied)")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise SourcePermissionError(str(path), f"could not stat file: {exc}") from exc
    if size == 0:
        raise EmptySourceError(str(path), "file is 0 bytes")


def _check_format_supported(path: Path) -> str:
    """Returns the lowercase suffix, or raises UnsupportedFormatError."""
    suffix = path.suffix.lower()
    if suffix in _KNOWN_UNSUPPORTED:
        raise UnsupportedFormatError(str(path), _KNOWN_UNSUPPORTED[suffix])
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFormatError(
            str(path),
            f"unsupported source format {suffix!r}; supported: {sorted(SUPPORTED_SUFFIXES)}",
        )
    return suffix


# --------------------------------------------------------------------------- #
# Stable source_id
# --------------------------------------------------------------------------- #


def _slugify(value: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug[:max_len] or "source"


def stable_source_id(identifier: str) -> str:
    """Deterministic source_id derived purely from the path or URL string.

    e.g. 'data/iso27001.pdf' -> 'iso27001-3f9a1c2b'

    Same input -> same id, every run, on every machine — that's the whole
    point, since downstream chunk IDs are derived from this. The slug keeps
    it human-readable; the 8-char sha256 suffix guarantees two different
    sources that happen to share a filename (two 'report.pdf' in different
    folders) never collide. If you need a plain filename-style id instead
    (e.g. 'iso27001.pdf'), pass `source_id=` explicitly to load_file —
    this is only the default.
    """
    if _is_url(identifier):
        parsed = urlparse(identifier)
        stem = parsed.path.rsplit("/", 1)[-1] or parsed.netloc
    else:
        stem = Path(identifier).stem
    digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:8]
    return f"{_slugify(stem)}-{digest}"


# --------------------------------------------------------------------------- #
# OCR helpers
# --------------------------------------------------------------------------- #


def _is_page_text_sufficient(text: str) -> bool:
    return len(text.strip()) >= _OCR_MIN_CHARS


def _ocr_pdf_page(path: Path, page_num: int) -> tuple[str, str]:
    """Render one PDF page to an image and OCR it.

    Returns (text, status) where status is one of:
      'ok'     - OCR produced text
      'empty'  - OCR ran cleanly but found nothing (genuinely blank page)
      'failed' - OCR deps missing, or rendering/OCR raised
    """
    global _ocr_deps_warned
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        if not _ocr_deps_warned:
            logger.warning(
                "OCR fallback unavailable: install pdf2image + pytesseract "
                "(and the poppler-utils + tesseract-ocr system packages) to "
                "OCR scanned PDF pages"
            )
            _ocr_deps_warned = True
        return "", "failed"

    try:
        images = convert_from_path(str(path), first_page=page_num, last_page=page_num, dpi=300)
        if not images:
            return "", "failed"
        text = pytesseract.image_to_string(images[0]) or ""
    except Exception as exc:
        logger.warning("OCR rendering/recognition failed for %s page %d: %s", path.name, page_num, exc)
        return "", "failed"

    return (text, "ok") if text.strip() else ("", "empty")


# --------------------------------------------------------------------------- #
# Per-format loaders
#
# All loaders share the signature (path, source_id, *, ocr=True) even though
# only PDF/image loaders use `ocr` — this keeps the dispatch table uniform.
#
# Every yielded RawDoc carries `extraction_status` (one of "ok" / "empty" /
# "failed" / "ocr") and `is_empty` in its metadata, so a genuinely blank page
# is distinguishable downstream from a page that failed to parse.
# --------------------------------------------------------------------------- #


def _load_pdf(path: Path, source_id: str, *, ocr: bool = True) -> Iterator[RawDoc]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - environment issue
        raise SourceError("pypdf is not installed; PDF sources cannot be loaded") from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise ParseError(str(path), f"PDF could not be opened: {exc}") from exc

    total = len(reader.pages)
    if total == 0:
        raise EmptySourceError(str(path), "PDF has zero pages")

    for i, page in enumerate(reader.pages):
        page_num = i + 1
        status = "ok"
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            # A single unparseable page must not abort a 400-page textbook.
            logger.warning(str(ParseError(str(path), f"text extraction failed: {exc}", page=page_num).failure))
            text = ""
            status = "failed"

        if status == "ok" and not _is_page_text_sufficient(text):
            # Near-zero extracted text usually means an image-only (scanned)
            # page rather than a genuinely blank one — try OCR before giving up.
            if ocr:
                ocr_text, ocr_status = _ocr_pdf_page(path, page_num)
                if ocr_status == "ok":
                    text, status = ocr_text, "ocr"
                elif ocr_status == "failed":
                    status = "failed"
                else:
                    status = "empty"
            else:
                status = "empty"

        yield RawDoc(
            text=text,
            source_id=source_id,
            metadata={
                "file_name": path.name,
                "source_id": source_id,
                "page_num": page_num,
                "page_count": total,
                "source_format": "pdf",
                "extraction_status": status,  # ok | empty | failed | ocr
                "is_empty": status == "empty",
            },
        )


def _load_text(path: Path, source_id: str, *, ocr: bool = True) -> Iterator[RawDoc]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Windows-authored files in this repo have shown up as UTF-16LE before.
        try:
            text = path.read_text(encoding="utf-16")
            logger.warning("%s was UTF-16 encoded, not UTF-8", path.name)
        except Exception as exc:
            raise ParseError(str(path), f"could not decode file as UTF-8 or UTF-16: {exc}") from exc
    except OSError as exc:
        raise SourcePermissionError(str(path), f"could not read file: {exc}") from exc

    status = "ok" if text.strip() else "empty"
    yield RawDoc(
        text=text,
        source_id=source_id,
        metadata={
            "file_name": path.name,
            "source_id": source_id,
            "page_num": 1,
            "page_count": 1,
            "source_format": path.suffix.lstrip("."),
            "extraction_status": status,
            "is_empty": status == "empty",
        },
    )


def _load_image(path: Path, source_id: str, *, ocr: bool = True) -> Iterator[RawDoc]:
    """png/jpg are always treated as raw/scanned — there's no "digital text"
    layer to try first, so this always OCRs.
    """
    try:
        from PIL import Image
        import pytesseract
    except ImportError as exc:
        raise SourceError("Pillow and pytesseract are required to load image sources") from exc

    try:
        image = Image.open(path)
        image.load()
    except Exception as exc:
        raise ParseError(str(path), f"image could not be opened: {exc}") from exc

    try:
        text = pytesseract.image_to_string(image) or ""
    except Exception as exc:
        raise ParseError(str(path), f"OCR failed: {exc}") from exc

    status = "ocr" if text.strip() else "empty"
    yield RawDoc(
        text=text,
        source_id=source_id,
        metadata={
            "file_name": path.name,
            "source_id": source_id,
            "page_num": 1,
            "page_count": 1,
            "source_format": path.suffix.lstrip(".").lower(),
            "extraction_status": status,
            "is_empty": status == "empty",
        },
    )


def _load_docx(path: Path, source_id: str, *, ocr: bool = True) -> Iterator[RawDoc]:
    """python-docx has no notion of rendered page breaks (that's a layout-time
    concept Word computes at render, not something stored in the file), so
    the whole document is one logical page. If per-page citations matter for
    a Word source, render it to PDF first and load that instead.
    """
    try:
        import docx
    except ImportError as exc:
        raise SourceError("python-docx is not installed; .docx sources cannot be loaded") from exc

    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise ParseError(str(path), f".docx could not be opened: {exc}") from exc

    text = "\n".join(p.text for p in document.paragraphs)
    status = "ok" if text.strip() else "empty"
    yield RawDoc(
        text=text,
        source_id=source_id,
        metadata={
            "file_name": path.name,
            "source_id": source_id,
            "page_num": 1,
            "page_count": 1,
            "source_format": "docx",
            "extraction_status": status,
            "is_empty": status == "empty",
        },
    )


def _extract_webpage_text(html: str, url: str) -> str:
    try:
        import trafilatura

        extracted = trafilatura.extract(html, url=url)
        if extracted:
            return extracted
    except ImportError:
        pass

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except ImportError as exc:
        raise SourceError(
            "neither trafilatura nor beautifulsoup4 is installed; cannot parse webpage HTML"
        ) from exc


def _load_webpage(url: str, source_id: str, *, ocr: bool = True) -> Iterator[RawDoc]:
    try:
        import requests
    except ImportError as exc:
        raise SourceError("requests is not installed; webpage sources cannot be loaded") from exc

    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 (loader-bot)"})
    except requests.RequestException as exc:
        raise SourceNotFoundError(url, f"could not reach URL: {exc}") from exc

    if resp.status_code in (401, 403):
        raise SourcePermissionError(url, f"HTTP {resp.status_code}: access denied")
    if resp.status_code == 404:
        raise SourceNotFoundError(url, "HTTP 404: page not found")
    if resp.status_code >= 400:
        raise ParseError(url, f"HTTP {resp.status_code} fetching page")

    html = resp.text
    if not html.strip():
        raise EmptySourceError(url, "page returned no content")

    text = _extract_webpage_text(html, url)
    status = "ok" if text.strip() else "empty"
    yield RawDoc(
        text=text,
        source_id=source_id,
        metadata={
            "file_name": url,
            "source_id": source_id,
            "page_num": 1,
            "page_count": 1,
            "source_format": "html",
            "extraction_status": status,
            "is_empty": status == "empty",
        },
    )


_DISPATCH = {
    ".pdf": _load_pdf,
    ".txt": _load_text,
    ".md": _load_text,
    ".png": _load_image,
    ".jpg": _load_image,
    ".jpeg": _load_image,
    ".docx": _load_docx,
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def load_file(
    path: Path | str,
    *,
    source_id: str | None = None,
    extra_metadata: dict | None = None,
    ocr: bool = True,
) -> list[RawDoc]:
    """Load one source — local file or http(s) URL — into RawDocs.

    Validates the source before attempting to parse it (exists/reachable,
    readable, supported format, non-empty) and raises a typed LoaderError
    subclass — SourceNotFoundError / SourcePermissionError /
    UnsupportedFormatError / EmptySourceError / ParseError — with structured
    source/page/reason context on failure. Never raises a bare stdlib
    exception like IndexError.

    `ocr` controls whether scanned PDF pages and raw images get an OCR pass
    (default True). Set False to skip OCR entirely (faster, but scanned
    pages will come back marked "empty" instead of "ocr").
    """
    raw = str(path)

    if _is_url(raw):
        sid = source_id or stable_source_id(raw)
        docs = list(_load_webpage(raw, sid, ocr=ocr))
    else:
        p = Path(path)
        _validate_local_source(p)
        suffix = _check_format_supported(p)
        sid = source_id or stable_source_id(str(p))
        loader_fn = _DISPATCH[suffix]
        try:
            docs = list(loader_fn(p, sid, ocr=ocr))
        except LoaderError:
            raise
        except Exception as exc:  # last-resort safety net — never leak a bare traceback
            raise ParseError(str(p), f"unexpected error while parsing: {exc}") from exc

    if not docs:
        raise EmptySourceError(raw, "no content could be extracted from source")

    if extra_metadata:
        for doc in docs:
            doc.metadata.update(extra_metadata)
    return docs


def load_directory(
    directory: Path | str,
    *,
    counts: StageCounts | None = None,
    recursive: bool = True,
    strict: bool = False,
    ocr: bool = True,
) -> list[RawDoc]:
    """Load every supported file under `directory`.

    `strict=False` (default) logs a structured "Loading failed: source=...
    reason=..." line and skips files that fail, so one corrupt PDF in a
    corpus of forty does not kill a twenty-minute ingestion run. Set
    `strict=True` for the user pipeline, where a single file failing means
    the user's upload failed and they need to be told.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise SourceNotFoundError(str(directory), "raw data directory not found")

    pattern = "**/*" if recursive else "*"
    files = sorted(
        p
        for p in directory.glob(pattern)
        if p.is_file() and (p.suffix.lower() in SUPPORTED_SUFFIXES or p.suffix.lower() in _KNOWN_UNSUPPORTED)
    )

    if not files:
        raise SourceNotFoundError(
            str(directory),
            f"no supported files found (looked for {sorted(SUPPORTED_SUFFIXES)})",
        )

    counts = counts if counts is not None else StageCounts()
    counts.files_seen = len(files)

    docs: list[RawDoc] = []
    for path in files:
        try:
            docs.extend(load_file(path, ocr=ocr))
        except LoaderError as exc:
            if strict:
                raise
            logger.error(str(exc.failure))
        except SourceError as exc:
            if strict:
                raise
            logger.error("skipping %s: %s", path.name, exc)

    counts.docs_loaded = len(docs)
    logger.info("loaded %d pages from %d files in %s", len(docs), len(files), directory)
    return docs


def load_urls(
    urls: Iterable[str],
    *,
    counts: StageCounts | None = None,
    strict: bool = False,
    extra_metadata: dict | None = None,
) -> list[RawDoc]:
    """Load a batch of webpage URLs — the URL analogue of load_directory.

    Same strict/non-strict skip-and-log behavior as load_directory.
    """
    urls = list(urls)
    if not urls:
        raise SourceNotFoundError("<empty url list>", "no URLs provided")

    counts = counts if counts is not None else StageCounts()
    counts.files_seen = len(urls)

    docs: list[RawDoc] = []
    for url in urls:
        try:
            docs.extend(load_file(url, extra_metadata=extra_metadata))
        except LoaderError as exc:
            if strict:
                raise
            logger.error(str(exc.failure))

    counts.docs_loaded = len(docs)
    logger.info("loaded %d pages from %d URLs", len(docs), len(urls))
    return docs