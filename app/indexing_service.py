"""
Indexing pipeline.

Responsibility:

Documents
    ↓
Cleaning
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Database
"""

from app.text_processor import clean_text
from app.text_processor import chunk_text

from app.embedding_service import embedding_service

from app.vector_store import vector_store


def index_documents(
    documents: list[str],
) -> int:
    """
    Process and index documents.

    Returns the total number of chunks stored.
    """

    all_chunks = []

    # Process every document.
    for document in documents:

        # STEP 1:
        # Clean document.
        cleaned_document = clean_text(document)

        # STEP 2:
        # Split document into chunks.
        chunks = chunk_text(
            cleaned_document,
            chunk_size=300,
            chunk_overlap=50,
        )

        # Add generated chunks to final list.
        all_chunks.extend(chunks)

    # Stop if no valid chunks exist.
    if not all_chunks:

        return 0

    # STEP 3:
    # Convert chunks into embeddings.
    embeddings = embedding_service.embed_documents(
        all_chunks
    )

    # STEP 4:
    # Store chunks + vectors.
    vector_store.add_documents(
        documents=all_chunks,
        embeddings=embeddings,
    )

    return len(all_chunks)