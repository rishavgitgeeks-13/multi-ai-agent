"""
Review Service
==============

Evaluates the content draft produced by the Writer Agent.

Input:
    draft          : str   — Markdown draft from WriterService
    strategy       : Dict  — full strategy (seo, keywords, tone, cta …)
    brand_context  : Dict  — tone, audience, pain_points, display_name
    revision_count : int   — how many rewrites have already happened

Output: Dict
    {
        "score"              : int,          # 0–100 weighted score
        "status"             : str,          # "PASS" | "FAIL"
        "needs_revision"     : bool,
        "feedback"           : List[str],    # positive observations
        "issues"             : List[str],    # problems found
        "rewrite_instruction": str,          # actionable brief for the Writer
        "dimension_scores"   : Dict[str, int],
        "revision_number"    : int,
    }

Evaluation dimensions
---------------------
  Content Quality    25 % — depth, clarity, value, factual grounding
  SEO Compliance     25 % — keyword density, headings, meta coverage
  Brand Alignment    20 % — tone match, audience fit, pain points addressed
  Structure          20 % — intro / body / conclusion, heading hierarchy
  CTA Effectiveness  10 % — clear, action-oriented, intent-matched

PASS threshold : score >= 70
"""

import json
import logging
import re
from typing import Dict, List, Tuple

from anthropic import Anthropic

from config.settings import settings

logger = logging.getLogger(__name__)

PASS_THRESHOLD = 70

DIMENSION_WEIGHTS: Dict[str, float] = {
    "content_quality": 0.25,
    "seo_compliance": 0.25,
    "brand_alignment": 0.20,
    "structure": 0.20,
    "cta_effectiveness": 0.10,
}


class ReviewService:
    """Evaluates content quality and returns a structured review decision."""

    def __init__(self) -> None:
        self._anthropic = Anthropic()
        self._model = settings.ANTHROPIC_MODEL
        self._temperature = 0.0     # reviews must be deterministic
        logger.info("ReviewService ready | model=%s", self._model)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        draft: str,
        strategy: Dict,
        brand_context: Dict,
        revision_count: int = 0,
    ) -> Dict:
        """Evaluate the draft and return a structured review decision."""
        logger.info(
            "ReviewService.run() | revision=%d | words=%d",
            revision_count,
            len(draft.split()),
        )

        # Rule-based pre-checks (fast, no LLM)
        pre_check_issues = self._run_pre_checks(draft, strategy)

        # LLM evaluation (all five dimensions in one call)
        llm_result = self._evaluate_via_llm(
            draft=draft,
            strategy=strategy,
            brand_context=brand_context,
            pre_check_issues=pre_check_issues,
        )

        # Weighted final score
        dim_scores = llm_result.get("dimension_scores", {})
        score = self._calculate_score(dim_scores)
        status = "PASS" if score >= PASS_THRESHOLD else "FAIL"
        needs_revision = status == "FAIL"

        review = {
            "score": score,
            "status": status,
            "needs_revision": needs_revision,
            "feedback": llm_result.get("feedback", []),
            "issues": pre_check_issues + llm_result.get("issues", []),
            "rewrite_instruction": (
                llm_result.get("rewrite_instruction", "")
                if needs_revision
                else ""
            ),
            "dimension_scores": dim_scores,
            "revision_number": revision_count + 1,
        }

        logger.info(
            "ReviewService complete | score=%d | status=%s",
            score,
            status,
        )
        return review

    # ------------------------------------------------------------------
    # Rule-based pre-checks
    # ------------------------------------------------------------------

    def _run_pre_checks(self, draft: str, strategy: Dict) -> List[str]:
        """
        Fast, rule-based checks that run before the LLM call.
        Returns a list of issue strings (empty = all passed).
        """
        issues: List[str] = []

        word_count = len(self._strip_markdown(draft).split())
        content_type = strategy.get("content_type", "article")

        # Word count check (skip for short-form types)
        if content_type in ("blog", "article"):
            if word_count < settings.MIN_ARTICLE_WORDS:
                issues.append(
                    f"Content too short: {word_count} words "
                    f"(minimum {settings.MIN_ARTICLE_WORDS})."
                )
            elif word_count > settings.MAX_ARTICLE_WORDS:
                issues.append(
                    f"Content too long: {word_count} words "
                    f"(maximum {settings.MAX_ARTICLE_WORDS})."
                )

        # Primary keyword presence check
        seo = strategy.get("seo", {})
        primary_keywords = seo.get("primary_keywords") or strategy.get("keywords", [])
        draft_lower = draft.lower()
        missing_keywords = [
            kw for kw in primary_keywords[:3]
            if kw.lower() not in draft_lower
        ]
        if missing_keywords:
            issues.append(
                f"Primary keywords not found in content: {', '.join(missing_keywords)}."
            )

        # Heading structure check
        h2_count = len(re.findall(r"^##\s+", draft, re.MULTILINE))
        if content_type in ("blog", "article") and h2_count < 2:
            issues.append(
                f"Insufficient headings: found {h2_count} H2 headings (minimum 2)."
            )

        # CTA check
        cta = strategy.get("cta") or brand_context_from_strategy(strategy)
        if cta and cta.lower() not in draft_lower:
            issues.append("CTA text not found in the content.")

        return issues

    # ------------------------------------------------------------------
    # LLM evaluation
    # ------------------------------------------------------------------

    def _evaluate_via_llm(
        self,
        draft: str,
        strategy: Dict,
        brand_context: Dict,
        pre_check_issues: List[str],
    ) -> Dict:
        """
        Run a single Anthropic call that scores all five dimensions
        and produces actionable feedback.
        """
        prompt = self._build_evaluation_prompt(
            draft=draft,
            strategy=strategy,
            brand_context=brand_context,
            pre_check_issues=pre_check_issues,
        )
        try:
            response = self._anthropic.messages.create(
                model=self._model,
                max_tokens=1024,
                temperature=self._temperature,
                system=(
                    "You are a senior content editor and SEO strategist. "
                    "You evaluate content objectively and give precise, actionable feedback. "
                    "Return valid JSON only — no prose, no markdown fences."
                ),
                messages=[{"role": "user", "content": prompt}],
            )
            return self._parse_evaluation(response.content[0].text)
        except Exception as exc:
            logger.error("ReviewService LLM call failed: %s — using fallback scores", exc)
            return self._fallback_evaluation(pre_check_issues)

    def _build_evaluation_prompt(
        self,
        draft: str,
        strategy: Dict,
        brand_context: Dict,
        pre_check_issues: List[str],
    ) -> str:
        """Build the structured evaluation prompt."""
        seo = strategy.get("seo", {})
        primary_kw = seo.get("primary_keywords") or strategy.get("keywords", [])
        secondary_kw = seo.get("secondary_keywords") or []
        tone = brand_context.get("tone") or strategy.get("tone", "professional")
        audience = brand_context.get("reader_segment") or strategy.get("audience", [])
        pain_points = brand_context.get("pain_points") or strategy.get("pain_points", [])
        cta = strategy.get("cta") or brand_context.get("cta", "")
        search_intent = seo.get("search_intent", "Informational")
        content_type = strategy.get("content_type", "article")

        audience_str = ", ".join(str(a) for a in audience) if isinstance(audience, list) else str(audience)
        pain_str = "; ".join(str(p) for p in pain_points[:4]) if pain_points else "none"
        primary_str = ", ".join(primary_kw[:5]) if primary_kw else "none"
        secondary_str = ", ".join(secondary_kw[:5]) if secondary_kw else "none"
        pre_issues_str = "\n".join(f"- {i}" for i in pre_check_issues) if pre_check_issues else "None"

        # Truncate draft to avoid token overflow
        truncated_draft = draft[:8000]
        if len(draft) > 8000:
            truncated_draft += "\n… [draft truncated]"

        return f"""Evaluate the following {content_type} draft.

=== EVALUATION CRITERIA ===
EXPECTED TONE        : {tone}
TARGET AUDIENCE      : {audience_str}
PAIN POINTS TO ADDRESS: {pain_str}
PRIMARY KEYWORDS     : {primary_str}
SECONDARY KEYWORDS   : {secondary_str}
SEARCH INTENT        : {search_intent}
CTA                  : {cta}

=== PRE-CHECK ISSUES (already identified) ===
{pre_issues_str}

=== DRAFT ===
{truncated_draft}

=== SCORING RUBRIC ===
Score each dimension 0–100:

content_quality (weight 25%)
  90–100: Exceptional depth, clear structure, compelling narrative
  70–89 : Good coverage, minor gaps in depth or clarity
  50–69 : Adequate but thin, lacks examples or data
  0–49  : Poor — vague, superficial, or off-topic

seo_compliance (weight 25%)
  90–100: Primary keywords in title, headings, and body; ideal density
  70–89 : Keywords mostly present; minor optimisation gaps
  50–69 : Keywords present but not well distributed
  0–49  : Keywords missing from headings or severely underused

brand_alignment (weight 20%)
  90–100: Tone, audience, and pain points perfectly addressed
  70–89 : Mostly aligned; minor tone or audience mismatch
  50–69 : Some misalignment in tone or audience targeting
  0–49  : Wrong tone, wrong audience, pain points not addressed

structure (weight 20%)
  90–100: Clear intro → body → conclusion, logical flow, good heading hierarchy
  70–89 : Good structure with minor flow issues
  50–69 : Structure present but transitions are weak
  0–49  : Poor structure — missing intro or conclusion, no logical progression

cta_effectiveness (weight 10%)
  90–100: Clear, specific, action-oriented CTA aligned with search intent
  70–89 : CTA present but could be stronger or more specific
  50–69 : Weak or vague CTA
  0–49  : No CTA or CTA misaligned with intent

=== TASK ===
Return ONLY this JSON object:
{{
  "dimension_scores": {{
    "content_quality": <int 0-100>,
    "seo_compliance": <int 0-100>,
    "brand_alignment": <int 0-100>,
    "structure": <int 0-100>,
    "cta_effectiveness": <int 0-100>
  }},
  "feedback": [
    "<specific positive observation>",
    "<specific positive observation>"
  ],
  "issues": [
    "<specific problem not already listed in pre-check issues>",
    "<specific problem>"
  ],
  "rewrite_instruction": "<If score < 70: one specific paragraph of actionable revision guidance for the Writer Agent. If score >= 70: empty string.>"
}}
"""

    def _parse_evaluation(self, raw: str) -> Dict:
        """Parse and validate the LLM evaluation JSON response."""
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        try:
            data = json.loads(cleaned)
            # Clamp scores to valid range
            dim_scores = data.get("dimension_scores", {})
            for key in DIMENSION_WEIGHTS:
                dim_scores[key] = max(0, min(100, int(dim_scores.get(key, 50))))
            data["dimension_scores"] = dim_scores
            return data
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Review JSON parse error: %s | raw=%s", exc, cleaned[:300])
            return self._fallback_evaluation([])

    def _fallback_evaluation(self, pre_check_issues: List[str]) -> Dict:
        """Return a conservative evaluation when the LLM call fails."""
        base_score = 50 if pre_check_issues else 65
        return {
            "dimension_scores": {d: base_score for d in DIMENSION_WEIGHTS},
            "feedback": ["Review service fallback — LLM evaluation unavailable."],
            "issues": pre_check_issues,
            "rewrite_instruction": (
                "Unable to generate specific instructions — please review the content manually."
                if pre_check_issues else ""
            ),
        }

    # ------------------------------------------------------------------
    # Score calculation
    # ------------------------------------------------------------------

    def _calculate_score(self, dimension_scores: Dict[str, int]) -> int:
        """Compute the weighted final score, rounded to the nearest integer."""
        weighted = sum(
            dimension_scores.get(dim, 0) * weight
            for dim, weight in DIMENSION_WEIGHTS.items()
        )
        return round(weighted)

    # ------------------------------------------------------------------
    # Markdown stripping utility
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Strip common Markdown syntax for plain-text word count."""
        text = re.sub(r"```[\s\S]*?```", " ", text)
        text = re.sub(r"`[^`]+`", " ", text)
        text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
        return text


def brand_context_from_strategy(strategy: Dict) -> str:
    """Extract CTA from nested brand context within strategy if present."""
    brand = strategy.get("brand_context", {})
    return str(brand.get("cta", "")) if isinstance(brand, dict) else ""
