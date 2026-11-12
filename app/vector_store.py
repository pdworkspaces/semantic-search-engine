"""
Vector database module.

Responsibility:

1. Connect to ChromaDB.
2. Create/get collection.
3. Store documents and embeddings.
4. Perform similarity search.
"""

import chromadb

from app.config import settings


class VectorStore:
    """
    Wrapper around ChromaDB.

    Other application modules should use this class
    instead of directly accessing ChromaDB.

    Benefits:

    1. Cleaner architecture.
    2. Easier testing.
    3. Easier database replacement later.
    """

    def __init__(self):

        # Create persistent ChromaDB client.
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory
        )

        # Get existing collection.

        # If it doesn't exist, Chroma creates it.
        self.collection = (
            self.client.get_or_create_collection(
                name=settings.collection_name,
                metadata={
                    "hnsw:space": "cosine"
                },
            )
        )

    def add_documents(
        self,
        documents: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """
        Store documents and embeddings.

        Each document requires a unique ID.
        """

        ids = [
            f"document-{index}"
            for index in range(len(documents))
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> dict:
        """
        Find vectors nearest to the query vector.

        Parameters
        ----------
        query_embedding:
            Vector representation of user query.

        top_k:
            Maximum results returned.

        Returns
        -------
        dict:
            Raw ChromaDB search results.
        """

        results = self.collection.query(

            # Chroma expects a list of query vectors.
            query_embeddings=[
                query_embedding
            ],

            n_results=top_k,

            include=[
                "documents",
                "distances",
            ],
        )

        return results


vector_store = VectorStore()