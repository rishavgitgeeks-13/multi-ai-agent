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
  8.  Classify search intent per keyword via heuristics (no LLM).
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
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI
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
    "semantic_similarity": 0.40,
    "tfidf": 0.20,
    "bm25": 0.15,
    "pain_point": 0.10,
    "search_intent": 0.10,
    "brand_relevance": 0.05,
}

# Maximum characters from corpus sent to the LLM to avoid token overflow.
_MAX_CORPUS_CHARS: int = 4_000

DEFAULT_LLM_MODEL: str = settings.OPENAI_MODEL


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
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not configured."
            )

        self._model = model
        self._temperature = temperature
        self._openai = OpenAI(
             api_key=settings.OPENAI_API_KEY
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

        # Workflows may prefix "[Brand: X]" for manager resolution — strip that
        # before keyword seeding/affinity so primaries stay searchable.
        clean_query = self._clean_user_query(user_input)

        # Step 1 — corpus (rank docs against the real topic, not the brand tag)
        corpus, documents = self._prepare_corpus(
            research_data,
            clean_query,
        )

        # Step 2 — seed from clean query + brand, then expand via LLM
        seeded = self._seed_candidates(clean_query, brand_context)
        llm_candidates = self._extract_candidate_keywords(
            corpus,
            clean_query,
            brand_context,
            seed_keywords=[c.keyword for c in seeded],
        )
        candidates = self._merge_candidates(seeded, llm_candidates)
        candidates = self._validate_against_evidence(
            candidates,
            corpus=corpus,
            user_input=clean_query,
            brand_context=brand_context,
        )
        logger.info(
            "Candidates ready | seeded=%d | llm=%d | validated=%d | clean_query=%s…",
            len(seeded),
            len(llm_candidates),
            len(candidates),
            clean_query[:60],
        )

        if not candidates:
            # Production fallback: never fail hard — use seeds or brand direction.
            candidates = seeded or self._seed_candidates(clean_query, brand_context)
            logger.warning(
                "Validation emptied candidates; falling back to %d seeds",
                len(candidates),
            )

        if not candidates:
            logger.warning("No candidates extracted; returning empty blueprint.")
            return self._empty_blueprint()

        # Steps 3–8 — all scoring dimensions
        candidates = self._score_tfidf(candidates, documents)
        candidates = self._score_bm25(candidates, documents)
        candidates = self._score_semantic_similarity(candidates, clean_query)
        candidates = self._score_pain_point(candidates, brand_context)
        candidates = self._score_brand_relevance(candidates, brand_context)
        candidates = self._classify_search_intent(candidates)

        # Step 9 — weighted final score
        candidates = self._calculate_final_scores(candidates)

        # Step 10 — rank
        ranked = self._rank_keywords(candidates)
        ranked = ranked[:20]

        # Step 11 — build blueprint (slots assigned by rank + query affinity)
        blueprint = self._build_blueprint(ranked, clean_query, brand_context)
        logger.info(
            "SEOService.run() complete | primary=%d | secondary=%d | intent=%s | primary=%s",
            len(blueprint.primary_keywords),
            len(blueprint.secondary_keywords),
            blueprint.search_intent,
            blueprint.primary_keywords,
        )
        return self._blueprint_to_dict(blueprint)

    # ------------------------------------------------------------------
    # Step 1 — Corpus preparation
    # ------------------------------------------------------------------

    def _prepare_corpus(
        self,
        research_data: Dict,
        user_input: str,
    ) -> Tuple[str, List[str]]:
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

        seen = set()
        unique_docs = []

        for doc in documents:
            key = doc[:500].lower().strip()
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        documents = unique_docs       

        documents = self._rank_documents_by_similarity(
            documents,
            user_input,
        )

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
        seed_keywords: Optional[List[str]] = None,
    ) -> List[KeywordCandidate]:
        """Call the LLM to extract structured keyword candidates from the corpus."""
        prompt = self._build_extraction_prompt(
            corpus,
            user_input,
            brand_context,
            seed_keywords=seed_keywords or [],
        )
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
        seed_keywords: Optional[List[str]] = None,
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
        seeds_str = ", ".join(seed_keywords) if seed_keywords else "none"

        return f"""You are performing keyword research for an SEO content strategy.

USER QUERY      : {user_input}
BRAND TONE      : {tone}
TARGET AUDIENCE : {rs_str}
KEYWORD DIRECTION: {kd_str}
SEED KEYWORDS   : {seeds_str}

RESEARCH CORPUS:
{truncated}

---

Expand and refine a focused keyword set strongly relevant to the user query.
Treat SEED KEYWORDS as required anchors (keep them or close variants).
Prioritize quality over quantity.
Target distribution:
- primary: 2-4 keywords (2–4 word core topic phrases closest to the user query)
- secondary: 5-8 keywords (supporting phrases; prefer ≤6 words)
- long_tail: 3-5 keywords
- industry: 2-4 keywords
- technical: 2-4 keywords

Return a JSON array — each object MUST follow this schema exactly:
{{
  "keyword": "<exact keyword phrase>",
  "category": "<primary|secondary|long_tail|industry|technical>"
}}

Category definitions:
  primary    – 2-4 word core topic keywords closest to the user query (never include workflow tags like "brand")
  secondary  – supporting keywords that complement primary topics
  long_tail  – 3-6 word specific phrases with clear search intent
  industry   – domain-specific jargon, terminology, acronyms
  technical  – technical terms, product names, frameworks, tools

Rules:
  - Only extract keywords supported by the corpus, user query, or seed list.
  - Primary keywords MUST be short searchable phrases (2–4 words), not full sentences.
  - Never invent keywords that start with the word "brand".
  - No duplicate keywords.
  - Lowercase all keywords.
  - Do not invent unrelated topics.
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
    # Seeding, merge, validation, slot assignment
    # ------------------------------------------------------------------

    _QUERY_STOPWORDS = frozenset({
        "a", "an", "the", "for", "to", "of", "in", "on", "and", "or",
        "how", "what", "why", "when", "where", "is", "are", "be", "with",
        "from", "by", "about", "into", "our", "your", "my", "we", "you",
        "write", "generate", "create", "make", "blog", "article", "post",
        "please", "need", "want",
        # Workflow-tag pollution — never treat as search terms
        "brand",
    })

    @staticmethod
    def _clean_user_query(user_input: str) -> str:
        """
        Strip workflow brand tags like '[Brand: Futuristix]' from the query.

        Does not change how brands are resolved elsewhere — only cleans the
        string used for keyword seeding and affinity scoring.
        """
        text = (user_input or "").strip()
        cleaned = re.sub(
            r"^\[Brand:\s*[^\]]+\]\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        return cleaned or text

    @staticmethod
    def _is_polluted_keyword(keyword: str) -> bool:
        """Reject brand leakage and ungrammatical SEO fragments."""
        kw = (keyword or "").strip().lower()
        if not kw:
            return True
        if kw == "brand" or kw.startswith("brand "):
            return True
        if re.search(r"\bgets?\s+\w+\s+people\b", kw):
            return True
        if re.search(r"\b\w+\s+gets?\s+scammed\s+people\b", kw):
            return True
        return False

    @staticmethod
    def _is_viable_primary(keyword: str) -> bool:
        """Primary slots must be short, searchable phrases (2–5 words)."""
        words = [w for w in (keyword or "").strip().split() if w]
        if len(words) < 2 or len(words) > 5:
            return False
        if words[0] == "brand":
            return False
        if SEOService._is_polluted_keyword(keyword):
            return False
        return True

    @staticmethod
    def _truncate_at_word(text: str, limit: int) -> str:
        """Hard length cap without cutting mid-word."""
        text = (text or "").strip()
        if len(text) <= limit:
            return text
        cut = text[: max(1, limit - 1)].rsplit(" ", 1)[0].rstrip(" ,;:-&|/")
        if not cut:
            cut = text[: limit - 1].rstrip(" ,;:-&|/")
        return cut + "…"

    def _seed_candidates(
        self,
        user_input: str,
        brand_context: Dict,
    ) -> List[KeywordCandidate]:
        """
        Deterministic seeds from the cleaned user query and brand keyword_direction.

        Prefers short searchable phrases (bigrams / 3–4 token cores) over dumping
        the whole query as one primary keyword.
        """
        seeds: List[KeywordCandidate] = []
        seen: set = set()

        clean = self._clean_user_query(user_input)
        tokens = [
            t for t in re.findall(r"[a-z0-9]+", clean.lower())
            if t not in self._QUERY_STOPWORDS and len(t) > 1
        ]

        def _add(phrase: str, category: str) -> None:
            phrase = phrase.strip().lower()
            if (
                not phrase
                or phrase in seen
                or self._is_polluted_keyword(phrase)
            ):
                return
            seen.add(phrase)
            seeds.append(KeywordCandidate(keyword=phrase, category=category))

        # Core topic phrases as primary seeds
        if len(tokens) >= 2:
            _add(" ".join(tokens[: min(4, len(tokens))]), "primary")
            _add(f"{tokens[0]} {tokens[1]}", "primary")
        if len(tokens) >= 3:
            _add(f"{tokens[0]} {tokens[1]} {tokens[2]}", "primary")
        if len(tokens) >= 4:
            # e.g. "ai agents … smb" style abbreviated topic
            _add(f"{tokens[0]} {tokens[1]} {tokens[-1]}", "secondary")

        for raw in brand_context.get("keyword_direction", []) or []:
            kw = str(raw).strip().lower()
            if not kw or kw in seen or self._is_polluted_keyword(kw):
                continue
            seen.add(kw)
            seeds.append(KeywordCandidate(keyword=kw, category="secondary"))

        return seeds

    @staticmethod
    def _merge_candidates(
        seeded: List[KeywordCandidate],
        llm_candidates: List[KeywordCandidate],
    ) -> List[KeywordCandidate]:
        """Deduplicate while preferring seed order first; drop polluted keywords."""
        merged: List[KeywordCandidate] = []
        seen: set = set()
        for candidate in list(seeded) + list(llm_candidates):
            key = candidate.keyword.strip().lower()
            if (
                not key
                or key in seen
                or SEOService._is_polluted_keyword(key)
            ):
                continue
            seen.add(key)
            merged.append(
                KeywordCandidate(
                    keyword=key,
                    category=candidate.category or "secondary",
                )
            )
        return merged

    def _validate_against_evidence(
        self,
        candidates: List[KeywordCandidate],
        corpus: str,
        user_input: str,
        brand_context: Dict,
    ) -> List[KeywordCandidate]:
        """
        Drop keywords that are unsupported by query, corpus, or brand seeds.

        If filtering would remove everything, return the original list unchanged
        so the pipeline never goes empty in production.
        """
        if not candidates:
            return []

        clean_query = self._clean_user_query(user_input)
        corpus_l = (corpus or "").lower()
        query_l = clean_query.lower()
        brand_allowed = {
            str(k).strip().lower()
            for k in (brand_context.get("keyword_direction") or [])
            if str(k).strip()
        }
        evidence = f"{query_l}\n{corpus_l}"

        valid: List[KeywordCandidate] = []
        for candidate in candidates:
            kw = candidate.keyword
            if self._is_polluted_keyword(kw):
                continue
            if kw in brand_allowed or kw in query_l or kw in corpus_l:
                valid.append(candidate)
                continue

            # Soft support: most content tokens appear in evidence.
            parts = [
                p for p in kw.split()
                if len(p) > 2 and p not in self._QUERY_STOPWORDS
            ]
            if parts and sum(1 for p in parts if p in evidence) >= max(1, len(parts) - 1):
                valid.append(candidate)
                continue

            logger.debug("Dropping unsupported keyword: %s", kw)

        if not valid:
            logger.warning(
                "Keyword validation removed all candidates — keeping originals"
            )
            return [
                c for c in candidates
                if not self._is_polluted_keyword(c.keyword)
            ] or candidates

        return valid

    @staticmethod
    def _query_affinity(keyword: str, user_input: str) -> float:
        """Fraction of keyword tokens that also appear in the cleaned user query."""
        stop = SEOService._QUERY_STOPWORDS
        clean = SEOService._clean_user_query(user_input)
        kw_tokens = {
            t for t in re.findall(r"[a-z0-9]+", (keyword or "").lower())
            if t not in stop and len(t) > 1
        }
        q_tokens = {
            t for t in re.findall(r"[a-z0-9]+", clean.lower())
            if t not in stop and len(t) > 1
        }
        if not kw_tokens or not q_tokens:
            return 0.0
        return len(kw_tokens & q_tokens) / float(len(kw_tokens))

    def _assign_keyword_slots(
        self,
        ranked: List[KeywordCandidate],
        user_input: str,
    ) -> Tuple[List[str], List[str]]:
        """
        Assign primary (max 2) and secondary (max 6) from ranked scores.

        Primary prefers short (2–5 word), high-affinity, high-score phrases.
        Long-tail phrases are kept for secondary — not forced into primary.
        """
        if not ranked:
            return [], []

        clean_query = self._clean_user_query(user_input)

        def _primary_fitness(c: KeywordCandidate) -> float:
            affinity = self._query_affinity(c.keyword, clean_query)
            score = float(c.final_score or 0.0)
            words = len(c.keyword.split())
            # Mild length preference: 2–4 words ideal for searchable primaries
            length_bonus = 0.15 if 2 <= words <= 4 else (0.05 if words == 5 else -0.25)
            viable_bonus = 0.2 if self._is_viable_primary(c.keyword) else -0.35
            return 0.50 * affinity + 0.35 * score + length_bonus + viable_bonus

        primary_pool = [
            c for c in ranked
            if self._is_viable_primary(c.keyword)
            and not self._is_polluted_keyword(c.keyword)
        ]
        if not primary_pool:
            primary_pool = [
                c for c in ranked
                if not self._is_polluted_keyword(c.keyword)
            ] or ranked

        primary_ranked = sorted(primary_pool, key=_primary_fitness, reverse=True)

        primary: List[str] = []
        for candidate in primary_ranked:
            if len(primary) >= 2:
                break
            affinity = self._query_affinity(candidate.keyword, clean_query)
            if not primary:
                primary.append(candidate.keyword)
            elif affinity > 0 or self._is_viable_primary(candidate.keyword):
                primary.append(candidate.keyword)

        if not primary and ranked:
            primary = [ranked[0].keyword]

        primary_set = set(primary)
        secondary: List[str] = []
        for candidate in ranked:
            if candidate.keyword in primary_set:
                continue
            if self._is_polluted_keyword(candidate.keyword):
                continue
            secondary.append(candidate.keyword)
            if len(secondary) >= 6:
                break

        # Reflect slots back onto candidates for keyword_scores consumers.
        for candidate in ranked:
            if candidate.keyword in primary_set:
                candidate.category = "primary"
            elif candidate.keyword in secondary:
                candidate.category = "secondary"

        return primary, secondary

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
        Rule-based search-intent classification (no LLM).

        Keeps SEO scoring intact while removing an entire LLM round-trip
        (and potential batch multiplies when many keywords are extracted).
        """
        for candidate in candidates:
            intent = self._heuristic_intent(candidate.keyword)
            candidate.search_intent = intent
            candidate.scores["search_intent"] = INTENT_SCORE_MAP.get(intent, 0.8)
        return candidates

    @staticmethod
    def _heuristic_intent(keyword: str) -> str:
        """Map a keyword to Informational / Commercial / Transactional / Navigational."""
        kw = (keyword or "").lower()

        if any(
            t in kw
            for t in (
                "buy",
                "pricing",
                "price",
                "order",
                "sign up",
                "signup",
                "download",
                "get started",
                "demo",
                "trial",
            )
        ):
            return "Transactional"

        if any(
            t in kw
            for t in (
                "best",
                " vs",
                "versus",
                "review",
                "compare",
                "comparison",
                "top ",
                "alternative",
            )
        ):
            return "Commercial"

        if any(
            t in kw
            for t in ("login", "official", "website", " portal", "dashboard")
        ):
            return "Navigational"

        return "Informational"

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
        primary, secondary = self._assign_keyword_slots(ranked, user_input)

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
            meta_title = self._truncate_at_word(
                str(data.get("meta_title", "")).strip(), 60
            )
            meta_description = self._truncate_at_word(
                str(data.get("meta_description", "")).strip(), 160
            )
            slug = str(data.get("slug", "")).strip()[:80] or self._slugify(user_input)

            # LLM sometimes returns blanks — never ship empty SEO fields.
            if not meta_title or not meta_description or not slug:
                fb_title, fb_desc, fb_slug = self._meta_fallback(
                    user_input, primary_keywords, brand_name
                )
                meta_title = meta_title or fb_title
                meta_description = meta_description or fb_desc
                slug = slug or fb_slug

            return (
                self._truncate_at_word(meta_title, 60),
                self._truncate_at_word(meta_description, 160),
                slug[:80],
            )

        except Exception as exc:
            logger.warning("Meta field generation failed: %s", exc)
            return self._meta_fallback(user_input, primary_keywords, brand_name)

    def _meta_fallback(
        self,
        user_input: str,
        primary_keywords: List[str],
        brand_name: str,
    ) -> Tuple[str, str, str]:
        """Rule-based meta title / description / slug when LLM output is unusable."""
        fallback_title = (
            f"{primary_keywords[0].title()} | {brand_name}"
            if primary_keywords
            else (user_input or "Untitled")
        )
        desc_src = user_input.strip() or fallback_title
        return (
            self._truncate_at_word(fallback_title, 60),
            self._truncate_at_word(desc_src, 160),
            self._slugify(user_input or fallback_title),
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _call_llm(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """Invoke the OpenAI model and return the plain text response."""
        response = self._openai.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (response.choices[0].message.content or "").strip()
    
    def _rank_documents_by_similarity(
        self,
        documents: List[str],
        user_input: str,
    ) -> List[str]:
        """
        Rank research documents by semantic similarity to the user query and
        keep only the most relevant ones to reduce token usage.
        """
        if len(documents) <= 10:
            return documents

        try:
            doc_embs = self._embed(documents)
            query_emb = self._embed([user_input])

            scores = (doc_embs @ query_emb.T).flatten()

            ranked = sorted(
                zip(documents, scores),
                key=lambda x: x[1],
                reverse=True,
            )

            top_docs = [
                doc
                for doc, _ in ranked[:10]
            ]

            logger.info(
                "Semantic document ranking: %d -> %d documents",
                len(documents),
                len(top_docs),
            )

            return top_docs

        except Exception as exc:
            logger.warning(
                "Document semantic ranking failed: %s",
                exc,
            )
            return documents

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
