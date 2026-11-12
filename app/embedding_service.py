"""
Embedding service.

Responsibility:

Convert text into numerical vectors.

We use Sentence Transformers.

Model:

all-MiniLM-L6-v2

Vector dimension:

384
"""

from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingService:
    """
    Service responsible for generating embeddings.

    The model is loaded once when the service
    object is created.

    Loading the model repeatedly would be slow
    and consume unnecessary memory.
    """

    def __init__(self):

        print(
            f"Loading embedding model: "
            f"{settings.embedding_model_name}"
        )

        # Load Sentence Transformer model.
        self.model = SentenceTransformer(
            settings.embedding_model_name
        )

    def embed_documents(
        self,
        documents: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.

        Parameters
        ----------
        documents:

            [
                "Python is used for AI",
                "React builds user interfaces"
            ]

        Returns
        -------
        list[list[float]]

        Example:

        [
            [0.12, 0.43, ...],
            [0.81, 0.11, ...]
        ]
        """

        # Convert documents into vectors.
        embeddings = self.model.encode(
            documents,
            normalize_embeddings=True,
        )

        # NumPy arrays are converted to Python lists
        # because databases/APIs commonly expect
        # serializable Python values.
        return embeddings.tolist()

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Generate an embedding for one search query.

        Parameters
        ----------
        query:

            "How does machine learning work?"

        Returns
        -------
        list[float]

            [0.12, 0.81, ...]
        """

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        return embedding.tolist()


# Create one shared service instance.

# This prevents loading the model repeatedly.
embedding_service = EmbeddingService()  