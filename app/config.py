"""
Application configuration.

This file contains configuration values used across
the entire semantic search application.

Keeping configuration separate makes the project
easier to maintain and modify.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Central configuration class.

    Every configuration value can later be loaded
    from environment variables or a .env file.
    """

    # Sentence Transformer model used to create embeddings.
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # Directory where ChromaDB stores vectors.
    chroma_persist_directory: str = "./chroma_db"

    # Name of the ChromaDB collection.
    collection_name: str = "documents"

    # Number of results returned by default.
    default_top_k: int = 3


# Create one settings object.

# Other modules can import this object instead
# of creating Settings repeatedly.
settings = Settings() 