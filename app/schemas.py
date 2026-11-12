"""
API request and response models.

Pydantic validates incoming API data.
"""

from pydantic import BaseModel
from pydantic import Field


class SearchRequest(BaseModel):
    """
    Request body for semantic search.
    """

    query: str = Field(
        min_length=1,
        description="Natural-language search query",
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum number of results",
    )


class SearchResult(BaseModel):
    """
    One semantic search result.
    """

    document: str

    distance: float

    similarity_score: float


class SearchResponse(BaseModel):
    """
    Complete search API response.
    """

    query: str

    total_results: int

    results: list[SearchResult]