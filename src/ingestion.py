"""
Document ingestion — PDF and plain text loading using PyPDF2 only.

All functions return a standardized dict:
    {
        "text":      str,   # full extracted text
        "filename":  str,   # basename of the source file
        "num_pages": int | None,
        "metadata": {
            "source":    str,  # absolute path
            "file_type": str,  # "pdf" | "txt"
            "pages":     int | None,
        }
    }

The "source" absolute path propagates through chunker metadata all the way
to the generator, enabling per-chunk source attribution in answers.
"""

from __future__ import annotations

from pathlib import Path

from src.utils import resolve_path

# ---------------------------------------------------------------------------
# PDF loading
# ---------------------------------------------------------------------------


def load_pdf(path: str) -> dict:
    """
    Load a PDF file and extract all text via PyPDF2.

    Args:
        path: Path to the PDF file.

    Returns:
        Standardized document dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the PDF contains no extractable text
                           (e.g. a scanned image PDF with no OCR layer).
        ImportError:       If PyPDF2 is not installed.
    """
    try:
        import PyPDF2
    except ImportError as exc:
        raise ImportError("PyPDF2 is required: pip install PyPDF2") from exc

    p = resolve_path(path)

    pages_text: list[str] = []
    with open(p, "rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        num_pages = len(reader.pages)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                pages_text.append(extracted)

    full_text = "\n\n".join(pages_text)
    if not full_text.strip():
        raise ValueError(
            f"No extractable text found in '{p}'. The PDF may be a scanned image without an OCR text layer."
        )

    return {
        "text": full_text,
        "filename": p.name,
        "num_pages": num_pages,
        "metadata": {
            "source": str(p),
            "file_type": "pdf",
            "pages": num_pages,
        },
    }


# ---------------------------------------------------------------------------
# Plain text loading
# ---------------------------------------------------------------------------


def load_text(path: str) -> dict:
    """
    Load a plain text (or Markdown) file.

    Args:
        path: Path to the text file.

    Returns:
        Standardized document dict.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = resolve_path(path)

    text = p.read_text(encoding="utf-8", errors="replace")

    return {
        "text": text,
        "filename": p.name,
        "num_pages": None,
        "metadata": {
            "source": str(p),
            "file_type": "txt",
            "pages": None,
        },
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def load_document(path: str) -> dict:
    """
    Load a document from disk, dispatching on file extension.

    Supported extensions: .pdf, .txt, .md

    Args:
        path: Path to the document.

    Returns:
        Standardized document dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the file extension is not supported.
    """
    suffix = Path(path).suffix.lower()

    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix in {".txt", ".md"}:
        return load_text(path)
    else:
        raise ValueError(f"Unsupported file type: '{suffix}'. Supported types are: .pdf, .txt, .md")
