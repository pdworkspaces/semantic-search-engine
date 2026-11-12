"""
Text processing module.

Responsibilities:

1. Clean text.
2. Split large documents into smaller chunks.

Chunking becomes extremely important when working
with PDFs, articles, documentation and RAG systems.
"""

import re


def clean_text(text: str) -> str:
    """
    Clean raw text.

    Current operations:

    1. Remove unnecessary whitespace.
    2. Remove repeated spaces.
    3. Remove leading/trailing spaces.

    Parameters
    ----------
    text:
        Raw document text.

    Returns
    -------
    str:
        Cleaned document.
    """

    # Replace multiple whitespace characters
    # with one space.
    text = re.sub(r"\s+", " ", text)

    # Remove spaces from beginning and end.
    text = text.strip()

    return text


def chunk_text(
    text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    Split a large document into overlapping chunks.

    Parameters
    ----------
    text:
        Complete document.

    chunk_size:
        Maximum characters in each chunk.

    chunk_overlap:
        Number of characters shared between
        consecutive chunks.

    Returns
    -------
    list[str]

    Example:

    Text:

    ABCDEFGHIJ

    chunk_size = 5
    overlap = 2

    Result:

    ABCDE
       DEFGH
          GHIJ
    """

    # Validate parameters.

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    chunks = []

    # Starting position.
    start = 0

    # Continue until complete document is processed.
    while start < len(text):

        # Calculate end position.
        end = start + chunk_size

        # Extract chunk.
        chunk = text[start:end]

        # Remove unnecessary spaces.
        chunk = chunk.strip()

        # Store non-empty chunks.
        if chunk:

            chunks.append(chunk)

        # Move forward.

        # We subtract overlap so some information
        # appears in consecutive chunks.
        start += chunk_size - chunk_overlap

    return chunks