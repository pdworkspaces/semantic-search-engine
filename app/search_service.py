"""
Semantic search service.

Responsibility:

User Query
    ↓
Query Embedding
    ↓
Vector Search
    ↓
Format Results
    ↓
Return Results
"""

from app.embedding_service import embedding_service
from app.vector_store import vector_store


def semantic_search(
    query: str,
    top_k: int,
) -> list[dict]:
    """
    Perform semantic search.

    Parameters
    ----------
    query:
        User's natural-language search query.

    top_k:
        Number of results requested.

    Returns
    -------
    list[dict]

    Example:

    [
        {
            "document": "...",
            "distance": 0.12,
            "similarity_score": 0.88
        }
    ]
    """

    # STEP 1:
    # Convert query into vector.
    query_embedding = embedding_service.embed_query(
        query
    )

    # STEP 2:
    # Search vector database.
    raw_results = vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k,
    )

    # Chroma returns nested lists because it supports
    # multiple query vectors at the same time.

    documents = raw_results["documents"][0]

    distances = raw_results["distances"][0]

    formatted_results = []

    # Combine each document with its distance.
    for document, distance in zip(
        documents,
        distances,
    ):

        # With cosine distance:
        #
        # similarity = 1 - distance
        #
        # Higher similarity means better match.

        similarity_score = 1 - distance

        formatted_results.append(
            {
                "document": document,
                "distance": distance,
                "similarity_score": similarity_score,
            }
        )

    return formatted_results