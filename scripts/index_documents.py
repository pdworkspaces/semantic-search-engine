"""
Command-line script used to index documents.

Run:

python -m scripts.index_documents
"""

from app.data_loader import load_text_documents
from app.indexing_service import index_documents


def main():
    """
    Execute indexing pipeline.
    """

    print("Loading documents...")

    documents = load_text_documents(
        "data/documents.txt"
    )

    print(
        f"Loaded {len(documents)} documents."
    )

    print("Creating embeddings and indexing...")

    total_chunks = index_documents(documents)

    print(
        f"Successfully indexed {total_chunks} chunks."
    )


if __name__ == "__main__":
    main()