"""
SEO Service
===========

Generates the SEO blueprint for the Strategy Agent.

Pipeline:
  1.  Merge all research documents into one corpus.
  2.  Extract candidate keywords via LLM (primary / secondary / long_tail /
      industry / technical).
  3.  Score each keyword with TF-IDF against the corpus.
  4.  Score each keyword with BM25 against the corpus.
  5.  Score each keyword with semantic cosine similarity vs. user query.
  6.  Score each keyword against brand pain points (cosine similarity).
  7.  Score each keyword against brand keyword direction (cosine similarity).
  8.  Classify search intent per keyword via LLM (batch).
  9.  Compute weighted final score.
  10. Rank keywords by final score descending.
  11. Build and return the SEO Blueprint.

Input:
    user_input      : str
    research_data   : Dict  (documents, sources, statistics, citations)
    brand_context   : Dict  (display_name, reader_segment, tone,
                             pain_points, keyword_direction)

Output:
    {
        "primary_keywords"  : List[str],
        "secondary_keywords": List[str],
        "keyword_scores"    : List[Dict],
        "search_intent"     : str,
        "meta_title"        : str,
        "meta_description"  : str,
        "slug"              : str,
    }

This service DOES NOT perform research.
It only analyzes the research package received from the Research Agent.
"""

import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from anthropic import Anthropic
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models.embeddings import embedding_provider
from config.settings import settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maps search intent label → numeric score used in weighted ranking.
INTENT_SCORE_MAP: Dict[str, float] = {
    "Commercial": 1.0,
    "Transactional": 0.9,
    "Informational": 0.8,
    "Navigational": 0.5,
}

# Weighted contribution of each scoring dimension to the final score.
SCORE_WEIGHTS: Dict[str, float] = {
    "semantic_similarity": 0.30,
    "tfidf": 0.20,
    "bm25": 0.20,
    "pain_point": 0.15,
    "search_intent": 0.10,
    "brand_relevance": 0.05,
}

# Maximum characters from corpus sent to the LLM to avoid token overflow.
_MAX_CORPUS_CHARS: int = 12_000

# Number of keywords per intent-classification batch (reduces LLM round-trips).
_INTENT_BATCH_SIZE: int = 20

DEFAULT_LLM_MODEL: str = settings.ANTHROPIC_MODEL


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class KeywordCandidate:
    """Holds a single keyword with its category, per-dimension scores, and metadata."""

    keyword: str
    category: str  # primary | secondary | long_tail | industry | technical
    scores: Dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0
    search_intent: str = ""


@dataclass
class SEOBlueprint:
    """Structured output produced by SEOService and consumed by the Strategy Agent."""

    primary_keywords: List[str]
    secondary_keywords: List[str]
    keyword_scores: List[Dict]
    search_intent: str
    meta_title: str
    meta_description: str
    slug: str


# ---------------------------------------------------------------------------
# SEOService
# ---------------------------------------------------------------------------


class SEOService:
    """
    Analyzes the research package and produces a ranked SEO Blueprint.

    Each public scoring concern lives in its own private method so the class
    remains testable and replaceable component by component.
    """

    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        temperature: float = 0.0,
    ) -> None:
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is not configured."
            )
            
        self._model = model
        self._temperature = temperature
        self._anthropic = Anthropic(      # reads ANTHROPIC_API_KEY from env
             api_key=settings.ANTHROPIC_API_KEY
        )
        logger.info(
            "SEOService ready | llm_model=%s | embedding_model=%s",
            model,
            "text-embedding-3-small",
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        user_input: str,
        research_data: Dict,
        brand_context: Dict,
    ) -> Dict:
        """Execute the full 11-step SEO pipeline and return the blueprint dict."""
        logger.info("SEOService.run() | query=%s…", user_input[:80])

        # Step 1 — corpus
        corpus, documents = self._prepare_corpus(research_data)

        # Step 2 — candidate keywords via LLM
        candidates = self._extract_candidate_keywords(corpus, user_input, brand_context)
        logger.info("Candidates extracted: %d", len(candidates))

        if not candidates:
            logger.warning("No candidates extracted; returning empty blueprint.")
            return self._empty_blueprint()

        # Steps 3–8 — all scoring dimensions
        candidates = self._score_tfidf(candidates, documents)
        candidates = self._score_bm25(candidates, documents)
        candidates = self._score_semantic_similarity(candidates, user_input)
        candidates = self._score_pain_point(candidates, brand_context)
        candidates = self._score_brand_relevance(candidates, brand_context)
        candidates = self._classify_search_intent(candidates)

        # Step 9 — weighted final score
        candidates = self._calculate_final_scores(candidates)

        # Step 10 — rank
        ranked = self._rank_keywords(candidates)

        # Step 11 — build blueprint
        blueprint = self._build_blueprint(ranked, user_input, brand_context)
        logger.info(
            "SEOService.run() complete | primary=%d | intent=%s",
            len(blueprint.primary_keywords),
            blueprint.search_intent,
        )
        return self._blueprint_to_dict(blueprint)

    # ------------------------------------------------------------------
    # Step 1 — Corpus preparation
    # ------------------------------------------------------------------

    def _prepare_corpus(self, research_data: Dict) -> Tuple[str, List[str]]:
        """
        Extract a flat document list and a single merged corpus string.

        Returns
        -------
        corpus    : str            – full merged text (used for LLM prompts)
        documents : List[str]      – individual documents (used for TF-IDF / BM25)
        """
        raw_docs: List = research_data.get("documents", [])
        documents: List[str] = []

        for doc in raw_docs:
            if isinstance(doc, str) and doc.strip():
                documents.append(doc.strip())
            elif isinstance(doc, dict):
                text = (
                    doc.get("text")
                    or doc.get("content")
                    or doc.get("body")
                    or ""
                )
                if str(text).strip():
                    documents.append(str(text).strip())

        # Append statistics and citations as additional signal documents.
        for stat in research_data.get("statistics", []):
            if isinstance(stat, str) and stat.strip():
                documents.append(stat.strip())
        for cite in research_data.get("citations", []):
            if isinstance(cite, str) and cite.strip():
                documents.append(cite.strip())

        corpus = "\n\n".join(documents)
        logger.debug(
            "Corpus: %d chars across %d documents", len(corpus), len(documents)
        )
        return corpus, documents

    # ------------------------------------------------------------------
    # Step 2 — LLM keyword extraction
    # ------------------------------------------------------------------

    def _extract_candidate_keywords(
        self,
        corpus: str,
        user_input: str,
        brand_context: Dict,
    ) -> List[KeywordCandidate]:
        """Call the LLM to extract structured keyword candidates from the corpus."""
        prompt = self._build_extraction_prompt(corpus, user_input, brand_context)
        try:
            raw = self._call_llm(
                system=(
                    "You are an expert SEO strategist. "
                    "Extract precise, high-value keywords from the research material. "
                    "Always respond with valid JSON only — no prose, no markdown fences."
                ),
                user=prompt,
            )

            logger.info(
                "Keyword extraction raw response:\n%s",
                raw[:3000],

            )
            candidates = self._parse_keyword_response(raw)

            logger.info(
                "Parsed keyword candidates: %d",
                len(candidates),
            )
            return candidates
        except Exception as exc:
            logger.error("Keyword extraction LLM call failed: %s", exc)
            return []

    def _build_extraction_prompt(
        self,
        corpus: str,
        user_input: str,
        brand_context: Dict,
    ) -> str:
        """Construct the structured keyword extraction prompt."""
        tone = brand_context.get("tone", "professional")
        keyword_direction = brand_context.get("keyword_direction", [])
        reader_segment = brand_context.get("reader_segment", [])

        truncated = corpus[:_MAX_CORPUS_CHARS]
        if len(corpus) > _MAX_CORPUS_CHARS:
            truncated += "\n… [corpus truncated for token limit]"

        kd_str = ", ".join(str(k) for k in keyword_direction) if keyword_direction else "none specified"
        rs_str = ", ".join(str(r) for r in reader_segment) if reader_segment else "general audience"

        return f"""You are performing keyword research for an SEO content strategy.

USER QUERY      : {user_input}
BRAND TONE      : {tone}
TARGET AUDIENCE : {rs_str}
KEYWORD DIRECTION: {kd_str}

RESEARCH CORPUS:
{truncated}

---

Extract 30–60 keywords that are strongly relevant to the user query and brand context.
Include a diverse mix of all five categories below.

Return a JSON array — each object MUST follow this schema exactly:
{{
  "keyword": "<exact keyword phrase>",
  "category": "<primary|secondary|long_tail|industry|technical>"
}}

Category definitions:
  primary    – 1-3 word core topic keywords (broad, high volume)
  secondary  – supporting keywords that complement primary topics
  long_tail  – 3-6 word specific phrases with clear search intent
  industry   – domain-specific jargon, terminology, acronyms
  technical  – technical terms, product names, frameworks, tools

Rules:
  - Only extract keywords supported by the corpus or user query.
  - No duplicate keywords.
  - Lowercase all keywords.
  - Return ONLY the JSON array — zero prose, zero markdown.
"""

    def _parse_keyword_response(self, raw: str) -> List[KeywordCandidate]:
        """Parse the LLM JSON response into a deduplicated KeywordCandidate list."""

        try:
            start = raw.find("[")
            end = raw.rfind("]")

            if start == -1 or end == -1:
                logger.error(
                    "No JSON array found in response:\\%s",
                    raw[:1000],
                )
                return []

            cleaned = raw[start:end + 1]
            items = json.loads(cleaned)

        except Exception as exc:
            logger.error(
                "Keyword JSON parse failed: %s\\nRaw:\\n%s",
                exc,
                raw[:1000],
            )
            return []

        candidates: List[KeywordCandidate] = []
        seen = set()

        for item in items:
            if not isinstance(item, dict):
                continue

            keyword = str(
                item.get("keyword", "")
            ).strip().lower()

            category = str(
                item.get("category", "secondary")
            ).strip().lower()

            if not keyword:
                continue

            if keyword in seen:
                continue

            seen.add(keyword)

            candidates.append(
                KeywordCandidate(
                    keyword=keyword,
                    category=category,
                )
            )

        return candidates

    # ------------------------------------------------------------------
    # Step 3 — TF-IDF score
    # ------------------------------------------------------------------

    def _score_tfidf(
        self,
        candidates: List[KeywordCandidate],
        documents: List[str],
    ) -> List[KeywordCandidate]:
        """
        Fit a TF-IDF vectorizer on the corpus documents, then measure each
        keyword's cosine similarity against the corpus centroid.
        Scores are min-max normalized to [0, 1].
        """
        if not documents:
            for c in candidates:
                c.scores["tfidf"] = 0.0
            return candidates

        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 3), lowercase=True)
            corpus_matrix = vectorizer.fit_transform(documents)

            keywords = [c.keyword for c in candidates]
            keyword_matrix = vectorizer.transform(keywords)

            # Corpus centroid = mean TF-IDF vector across all documents.
            centroid = np.asarray(corpus_matrix.mean(axis=0))  # shape (1, vocab)
            raw_scores = cosine_similarity(keyword_matrix, centroid).flatten()
            normalized = self._normalize_array(raw_scores)

            for candidate, score in zip(candidates, normalized):
                candidate.scores["tfidf"] = float(score)

        except Exception as exc:
            logger.warning("TF-IDF scoring failed: %s", exc)
            for c in candidates:
                c.scores["tfidf"] = 0.0

        return candidates

    # ------------------------------------------------------------------
    # Step 4 — BM25 score
    # ------------------------------------------------------------------

    def _score_bm25(
        self,
        candidates: List[KeywordCandidate],
        documents: List[str],
    ) -> List[KeywordCandidate]:
        """
        Build a BM25Okapi index over the corpus documents and score each keyword.
        The max BM25 score across all documents is used per keyword (best-match
        semantics), then normalized to [0, 1].
        """
        if not documents:
            for c in candidates:
                c.scores["bm25"] = 0.0
            return candidates

        try:
            tokenized = [doc.lower().split() for doc in documents]
            bm25 = BM25Okapi(tokenized)

            raw_scores = np.array(
                [
                    float(np.max(bm25.get_scores(c.keyword.lower().split())))
                    for c in candidates
                ]
            )
            normalized = self._normalize_array(raw_scores)

            for candidate, score in zip(candidates, normalized):
                candidate.scores["bm25"] = float(score)

        except Exception as exc:
            logger.warning("BM25 scoring failed: %s", exc)
            for c in candidates:
                c.scores["bm25"] = 0.0

        return candidates

    # ------------------------------------------------------------------
    # Step 5 — Semantic similarity (keyword vs user query)
    # ------------------------------------------------------------------

    def _score_semantic_similarity(
        self,
        candidates: List[KeywordCandidate],
        user_input: str,
    ) -> List[KeywordCandidate]:
        """
        Embed the user query and all keywords with the OpenAI embedding model.
        Cosine similarity (dot product of L2-normalized vectors) is returned
        directly in [0, 1] — no further normalization needed.
        """
        try:
            query_emb = self._embed([user_input])                        # (1, dim)
            kw_embs = self._embed([c.keyword for c in candidates])       # (n, dim)

            # dot product of L2-normalized vectors = cosine similarity
            sims = np.clip((kw_embs @ query_emb.T).flatten(), 0.0, 1.0)

            for candidate, score in zip(candidates, sims):
                candidate.scores["semantic_similarity"] = float(score)

        except Exception as exc:
            logger.warning("Semantic similarity scoring failed: %s", exc)
            for c in candidates:
                c.scores["semantic_similarity"] = 0.0

        return candidates

    # ------------------------------------------------------------------
    # Step 6 — Pain point score
    # ------------------------------------------------------------------

    def _score_pain_point(
        self,
        candidates: List[KeywordCandidate],
        brand_context: Dict,
    ) -> List[KeywordCandidate]:
        """
        Measure cosine similarity between each keyword and the concatenated
        brand pain points string.  Returns 0.0 when pain points are absent.
        """
        pain_points: List = brand_context.get("pain_points", [])

        if not pain_points:
            for c in candidates:
                c.scores["pain_point"] = 0.0
            return candidates

        try:
            pain_text = " ".join(str(p) for p in pain_points)
            pain_emb = self._embed([pain_text])                          # (1, dim)
            kw_embs = self._embed([c.keyword for c in candidates])       # (n, dim)
            sims = np.clip((kw_embs @ pain_emb.T).flatten(), 0.0, 1.0)

            for candidate, score in zip(candidates, sims):
                candidate.scores["pain_point"] = float(score)

        except Exception as exc:
            logger.warning("Pain point scoring failed: %s", exc)
            for c in candidates:
                c.scores["pain_point"] = 0.0

        return candidates

    # ------------------------------------------------------------------
    # Step 7 — Brand relevance score
    # ------------------------------------------------------------------

    def _score_brand_relevance(
        self,
        candidates: List[KeywordCandidate],
        brand_context: Dict,
    ) -> List[KeywordCandidate]:
        """
        Measure cosine similarity between each keyword and the brand's stated
        keyword direction.  Returns 0.0 when keyword direction is absent.
        """
        keyword_direction: List = brand_context.get("keyword_direction", [])

        if not keyword_direction:
            for c in candidates:
                c.scores["brand_relevance"] = 0.0
            return candidates

        try:
            direction_text = " ".join(str(k) for k in keyword_direction)
            dir_emb = self._embed([direction_text])                      # (1, dim)
            kw_embs = self._embed([c.keyword for c in candidates])       # (n, dim)
            sims = np.clip((kw_embs @ dir_emb.T).flatten(), 0.0, 1.0)

            for candidate, score in zip(candidates, sims):
                candidate.scores["brand_relevance"] = float(score)

        except Exception as exc:
            logger.warning("Brand relevance scoring failed: %s", exc)
            for c in candidates:
                c.scores["brand_relevance"] = 0.0

        return candidates

    # ------------------------------------------------------------------
    # Step 8 — Search intent classification
    # ------------------------------------------------------------------

    def _classify_search_intent(
        self,
        candidates: List[KeywordCandidate],
    ) -> List[KeywordCandidate]:
        """
        Batch-classify search intent for all candidates.
        Batching into groups of _INTENT_BATCH_SIZE reduces LLM round-trips.
        """
        for i in range(0, len(candidates), _INTENT_BATCH_SIZE):
            batch = candidates[i : i + _INTENT_BATCH_SIZE]
            self._classify_intent_batch(batch)
        return candidates

    def _classify_intent_batch(
        self,
        batch: List[KeywordCandidate],
    ) -> None:
        """
        Classify search intent for one batch of keywords in a single LLM call.
        Modifies each KeywordCandidate in-place.
        """
        keyword_lines = "\n".join(
            f"{idx + 1}. {c.keyword}" for idx, c in enumerate(batch)
        )
        prompt = f"""Classify the search intent of each keyword below.

Valid intents (choose exactly one per keyword):
  Informational  – user wants to learn (how, what, why, guide, tips, explained)
  Commercial     – user is researching before buying (best, vs, review, compare, top)
  Transactional  – user is ready to act (buy, pricing, order, sign up, download, get)
  Navigational   – user seeks a specific site or brand (brand name, login, official)

Keywords:
{keyword_lines}

Return ONLY a JSON array with one entry per keyword in the SAME order:
[
  {{"idx": 1, "intent": "Commercial"}},
  ...
]
No prose. No markdown.
"""
        try:
            raw = self._call_llm(
                system="You are an SEO expert. Return valid JSON only.",
                user=prompt,
            )
            cleaned = (
                re.sub(r"```(?:json)?", "", raw)
                .strip()
                .strip("`")
                .strip()
            )
            classifications = json.loads(cleaned)
            intent_map: Dict[int, str] = {
                item["idx"]: item["intent"] for item in classifications
            }

            for idx, candidate in enumerate(batch):
                intent = intent_map.get(idx + 1, "Informational")
                candidate.search_intent = intent
                candidate.scores["search_intent"] = INTENT_SCORE_MAP.get(intent, 0.8)

        except Exception as exc:
            logger.warning("Intent classification batch failed: %s", exc)
            for candidate in batch:
                candidate.search_intent = "Informational"
                candidate.scores["search_intent"] = INTENT_SCORE_MAP["Informational"]

    # ------------------------------------------------------------------
    # Step 9 — Weighted final score
    # ------------------------------------------------------------------

    def _calculate_final_scores(
        self,
        candidates: List[KeywordCandidate],
    ) -> List[KeywordCandidate]:
        """Compute the weighted sum of all dimension scores for every candidate."""
        for candidate in candidates:
            final = sum(
                SCORE_WEIGHTS[dim] * candidate.scores.get(dim, 0.0)
                for dim in SCORE_WEIGHTS
            )
            candidate.final_score = round(float(final), 6)
        return candidates

    # ------------------------------------------------------------------
    # Step 10 — Rank
    # ------------------------------------------------------------------

    def _rank_keywords(
        self,
        candidates: List[KeywordCandidate],
    ) -> List[KeywordCandidate]:
        """Return candidates sorted by final_score descending."""
        return sorted(candidates, key=lambda c: c.final_score, reverse=True)

    # ------------------------------------------------------------------
    # Step 11 — SEO Blueprint
    # ------------------------------------------------------------------

    def _build_blueprint(
        self,
        ranked: List[KeywordCandidate],
        user_input: str,
        brand_context: Dict,
    ) -> SEOBlueprint:
        """Assemble the final SEO Blueprint from ranked keyword candidates."""
        primary = [c.keyword for c in ranked if c.category == "primary"][:5]
        secondary = [
            c.keyword
            for c in ranked
            if c.category in ("secondary", "long_tail")
        ][:10]

        # Dominant intent = most frequent intent label among the top-10 keywords.
        top_intents = [c.search_intent for c in ranked[:10] if c.search_intent]
        dominant_intent = (
            max(set(top_intents), key=top_intents.count)
            if top_intents
            else "Informational"
        )

        meta_title, meta_description, slug = self._generate_meta_fields(
            primary_keywords=primary,
            user_input=user_input,
            brand_context=brand_context,
            dominant_intent=dominant_intent,
        )

        return SEOBlueprint(
            primary_keywords=primary,
            secondary_keywords=secondary,
            keyword_scores=[self._candidate_to_dict(c) for c in ranked],
            search_intent=dominant_intent,
            meta_title=meta_title,
            meta_description=meta_description,
            slug=slug,
        )

    def _generate_meta_fields(
        self,
        primary_keywords: List[str],
        user_input: str,
        brand_context: Dict,
        dominant_intent: str,
    ) -> Tuple[str, str, str]:
        """
        Use the LLM to generate an SEO meta title, meta description, and URL slug
        that reflect the top keywords, brand identity, and dominant search intent.
        Falls back to rule-based construction on any failure.
        """
        brand_name = brand_context.get("display_name", "")
        tone = brand_context.get("tone", "professional")
        top_kw_str = ", ".join(primary_keywords[:3]) if primary_keywords else user_input

        prompt = f"""Generate SEO meta fields for a content piece.

User Query     : {user_input}
Brand          : {brand_name}
Tone           : {tone}
Search Intent  : {dominant_intent}
Top Keywords   : {top_kw_str}

Requirements:
  meta_title       – 50–60 characters. Lead with the primary keyword.
                     Include brand name only if it fits within the limit.
  meta_description – 150–160 characters. Benefit-driven, ends with a CTA
                     appropriate for the search intent.
  slug             – Lowercase, hyphens only, 3–6 words, keyword-rich.
                     No stop words (a, the, of, …) unless essential for meaning.

Return ONLY this JSON object — no prose, no markdown:
{{
  "meta_title": "...",
  "meta_description": "...",
  "slug": "..."
}}
"""
        try:
            raw = self._call_llm(
                system="You are an expert SEO copywriter. Return valid JSON only.",
                user=prompt,
            )
            cleaned = (
                re.sub(r"```(?:json)?", "", raw)
                .strip()
                .strip("`")
                .strip()
            )
            data = json.loads(cleaned)
            return (
                str(data.get("meta_title", ""))[:60],
                str(data.get("meta_description", ""))[:160],
                str(data.get("slug", self._slugify(user_input)))[:80],
            )

        except Exception as exc:
            logger.warning("Meta field generation failed: %s", exc)
            fallback_title = (
                f"{primary_keywords[0].title()} | {brand_name}"
                if primary_keywords
                else user_input[:60]
            )
            return (
                fallback_title[:60],
                (user_input[:155] + "…") if len(user_input) > 155 else user_input,
                self._slugify(user_input),
            )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _call_llm(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """Invoke the Anthropic model and return the plain text response."""
        response = self._anthropic.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=self._temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    def _embed(self, texts: List[str]) -> np.ndarray:
        """
        Return an L2-normalized embedding matrix via the shared EmbeddingProvider.

        Shape  : (len(texts), 1536)
        Each row is unit-length: dot_product(row_i, row_j) == cosine_similarity.
        """
        return embedding_provider.embed(texts)

    @staticmethod
    def _normalize_array(arr: np.ndarray) -> np.ndarray:
        """Min-max normalize a 1-D array to [0, 1]. Returns zeros when range ≈ 0."""
        arr = arr.astype(float)
        lo, hi = arr.min(), arr.max()
        if (hi - lo) < 1e-9:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert arbitrary text to a lowercase, hyphen-separated URL slug."""
        # Normalize unicode → ASCII
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        text = re.sub(r"[^\w\s-]", "", text.lower())
        text = re.sub(r"[\s_]+", "-", text).strip("-")
        # Remove consecutive hyphens
        text = re.sub(r"-{2,}", "-", text)
        return text[:80]

    @staticmethod
    def _candidate_to_dict(candidate: KeywordCandidate) -> Dict:
        """Serialize a KeywordCandidate to a plain dict for JSON output."""
        return {
            "keyword": candidate.keyword,
            "category": candidate.category,
            "search_intent": candidate.search_intent,
            "scores": {k: round(v, 4) for k, v in candidate.scores.items()},
            "final_score": round(candidate.final_score, 4),
        }

    @staticmethod
    def _blueprint_to_dict(blueprint: SEOBlueprint) -> Dict:
        """Serialize SEOBlueprint to the contract Dict consumed by Strategy Agent."""
        return {
            "primary_keywords": blueprint.primary_keywords,
            "secondary_keywords": blueprint.secondary_keywords,
            "keyword_scores": blueprint.keyword_scores,
            "search_intent": blueprint.search_intent,
            "meta_title": blueprint.meta_title,
            "meta_description": blueprint.meta_description,
            "slug": blueprint.slug,
        }

    @staticmethod
    def _empty_blueprint() -> Dict:
        """Return a safe empty blueprint when no candidates could be extracted."""
        return {
            "primary_keywords": [],
            "secondary_keywords": [],
            "keyword_scores": [],
            "search_intent": "Informational",
            "meta_title": "",
            "meta_description": "",
            "slug": "",
        }


# ---------------------------------------------------------------------------
# Quick smoke-test (requires OPENAI_API_KEY in environment)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint

    research_data = {
        "documents": [
            (
                "Generative AI is transforming B2B SaaS marketing. "
                "Companies using AI-driven content automation report 40% faster "
                "content production and 25% higher conversion rates. Tools like "
                "GPT-4, Claude, and Gemini are being integrated into CMS platforms "
                "to auto-generate SEO-optimized blog posts, landing pages, and emails."
            ),
            (
                "ROI measurement for AI marketing tools remains a challenge. "
                "Marketers cite attribution complexity and integration costs as the "
                "top barriers to adoption. However, early adopters show a 3x increase "
                "in organic search traffic within six months of deploying AI content "
                "workflows."
            ),
            (
                "LinkedIn and Reddit communities discuss prompt engineering as a "
                "critical skill for content teams. Long-form guides, comparison "
                "articles, and case studies rank highest for commercial intent keywords "
                "in the SaaS space."
            ),
        ],
        "sources": ["TechCrunch", "Search Engine Journal", "G2"],
        "statistics": [
            "40% faster content production with AI tools (McKinsey, 2024)",
            "3x organic traffic lift within 6 months for early AI adopters",
        ],
        "citations": [
            "McKinsey Global AI Report 2024",
            "HubSpot State of Marketing 2024",
        ],
    }

    brand_context = {
        "display_name": "Futuristix",
        "reader_segment": ["B2B SaaS founders", "content marketers", "growth hackers"],
        "tone": "ROI Driven",
        "pain_points": [
            "too slow to produce content",
            "low organic traffic",
            "poor conversion from blog posts",
            "can't justify AI tool budget to leadership",
        ],
        "keyword_direction": [
            "AI content automation",
            "SaaS marketing ROI",
            "organic growth",
            "GPT for marketing",
        ],
    }

    service = SEOService(model=DEFAULT_LLM_MODEL)
    blueprint = service.run(
        user_input="How to use AI to automate B2B SaaS content marketing and improve ROI",
        research_data=research_data,
        brand_context=brand_context,
    )

    pprint.pprint(blueprint)
