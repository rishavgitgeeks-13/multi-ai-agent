"""
Research Schema
===============

Pydantic models for the data contract produced by the Research Agent
and consumed by the Strategy Agent.

ResearchService.run() must return data that is valid against ResearchData.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ==========================================================================
# Sub-models
# ==========================================================================


class ResearchSource(BaseModel):
    """One external or internal source referenced in the research package."""

    title: str = ""
    url: str = ""
    source_type: str = ""
    """
    web      – Tavily / DuckDuckGo web search result
    kb       – internal brand knowledge base hit
    news     – Google News RSS / NewsAPI article
    youtube  – YouTube transcript snippet
    reddit   – Reddit discussion thread
    """
    published_date: Optional[str] = None
    author: Optional[str] = None
    snippet: str = ""


class ResearchDocument(BaseModel):
    """
    One retrieved document (KB chunk or web result) with its provenance.
    The `text` field is the raw content consumed by TF-IDF / BM25 / LLM.
    """

    text: str
    title: str = ""
    url: str = ""
    source_type: str = ""
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ==========================================================================
# Top-level Research Data contract
# ==========================================================================


class ResearchData(BaseModel):
    """
    Full research package produced by ResearchService and stored in
    ContentState["research_data"].

    The Strategy Agent (SEOService, HashtagService) reads from this model.
    """

    documents: List[ResearchDocument] = Field(default_factory=list)
    total_documents: int = 0

    sources: List[ResearchSource] = Field(default_factory=list)
    """Deduplicated list of all sources referenced by `documents`."""

    statistics: List[str] = Field(default_factory=list)
    """
    Factual data points extracted from research (e.g. '40% faster with AI').
    Injected into Writer prompts to ground the content.
    """

    citations: List[str] = Field(default_factory=list)
    """
    Formatted citation strings (e.g. 'McKinsey Global AI Report 2024').
    Appear in the citation section of the final content.
    """

    def to_state_dict(self) -> Dict[str, Any]:
        """
        Serialize to the flat dict expected by ContentState["research_data"].
        Converts ResearchDocument objects to plain dicts for JSON serialization.
        """
        return {
            "documents": [doc.model_dump() for doc in self.documents],
            "total_documents": self.total_documents,
            "sources": [src.model_dump() for src in self.sources],
            "statistics": self.statistics,
            "citations": self.citations,
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "ResearchData":
        """Reconstruct from ContentState["research_data"]."""
        return cls(**data)
