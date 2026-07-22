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

PASS threshold : score >= 95
"""

import json
import logging
import re
from typing import Dict, List, Tuple

from openai import OpenAI
from config.settings import settings

logger = logging.getLogger(__name__)

PASS_THRESHOLD = 95

DIMENSION_WEIGHTS: Dict[str, float] = {
    "content_quality": 0.18,
    "seo_compliance": 0.22,
    "brand_alignment": 0.18,
    "structure": 0.12,
    "factual_grounding": 0.15,
    "natural_voice": 0.10,
    "cta_effectiveness": 0.05,
}

# Phrases that make content read as AI-generated. Flagged in pre-checks and
# penalised in the natural_voice dimension.
AI_TELL_PHRASES: List[str] = [
    "in today's fast-paced world",
    "in today's digital age",
    "in the ever-evolving",
    "in the world of",
    "when it comes to",
    "it's worth noting",
    "it is worth noting",
    "it's important to note",
    "it is important to note",
    "needless to say",
    "at the end of the day",
    "in conclusion",
    "in summary",
    "to sum up",
    "moreover",
    "furthermore",
    "additionally,",
    "however, it is",
    "as we can see",
    "in this article, we will",
    "in this article, we'll",
    "this article will",
    "let's dive in",
    "let's dive into",
    "dive deep",
    "unlock the power",
    "unleash the",
    "in the realm of",
    "navigating the",
    "a game-changer",
    "game changer",
    "the key takeaway",
    "rest assured",
    "look no further",
    "we've got you covered",
    "whether you're",
    "not only ... but also",
    "plays a crucial role",
    "plays a vital role",
    "plays a pivotal role",
    "a testament to",
    "in essence",
    "ultimately,",
    "elevate your",
    "in the fast-paced",
    "ever-changing landscape",
]


class ReviewService:
    """Evaluates content quality and returns a structured review decision."""

    def __init__(self) -> None:
        # Ensure OpenAI credentials are available
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not configured."
            )

        # Create authenticated OpenAI client
        self._openai = OpenAI(
            api_key=settings.OPENAI_API_KEY
        )

        # Review configuration
        self._model = settings.OPENAI_MODEL
        self._temperature = 0.0  # deterministic reviews

        logger.info(
            "ReviewService ready | model=%s",
            self._model,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        draft: str,
        strategy: Dict,
        brand_context: Dict,
        revision_count: int = 0,
        primary_topic: str = "",
        user_input: str = "",
    ) -> Dict:
        """Evaluate the draft and return a structured review decision."""
        logger.info(
            "ReviewService.run() | revision=%d | words=%d",
            revision_count,
            len((draft or "").split()),
        )

        brief = (primary_topic or user_input or "").strip()

        # Hard fail empty drafts immediately (no LLM spend on blank content).
        if not (draft or "").strip():
            content_type = str(strategy.get("content_type", "article")).lower()
            secondary_bit = (
                " Place at least one secondary keyword in the introduction and one "
                "in the conclusion."
                if content_type in ("blog", "article")
                else " Do not keyword-stuff; keep short-form copy natural."
            )
            return {
                "score": 0,
                "status": "FAIL",
                "needs_revision": True,
                "feedback": [],
                "issues": [
                    "Draft is empty (0 words). Regenerate the full article from scratch."
                ],
                "rewrite_instruction": (
                    "The previous draft was empty. Write the COMPLETE piece from "
                    "scratch following the outline and target length. Return full "
                    "Markdown only — never return an empty response. Use a natural "
                    "human voice (no Moreover/Furthermore/In conclusion)."
                    + secondary_bit
                    + (
                        f" Stay on this primary brief: {brief[:240]}"
                        if brief
                        else ""
                    )
                ),
                "dimension_scores": {d: 0 for d in DIMENSION_WEIGHTS},
                "revision_number": revision_count + 1,
            }

        # Rule-based pre-checks (fast, no LLM)
        pre_check_issues = self._run_pre_checks(draft, strategy)
        if brief:
            fidelity_issue = self._topic_fidelity_issue(brief, draft)
            if fidelity_issue:
                pre_check_issues.append(fidelity_issue)

        # LLM evaluation (all dimensions in one call)
        llm_result = self._evaluate_via_llm(
            draft=draft,
            strategy=strategy,
            brand_context=brand_context,
            pre_check_issues=pre_check_issues,
            primary_topic=brief,
        )

        # Weighted final score
        dim_scores = llm_result.get("dimension_scores", {})
        # Cap natural_voice when AI-cliché pre-check fired
        found_tells = self._detect_ai_tells(draft.lower())
        if found_tells:
            dim_scores["natural_voice"] = min(
                int(dim_scores.get("natural_voice", 50)),
                60,
            )

        # Off-brief drafts must not pass with a high content_quality score.
        fidelity_flags = " ".join(str(i).lower() for i in pre_check_issues)
        if any(
            flag in fidelity_flags
            for flag in (
                "off-brief",
                "india geography",
                "year-range",
                "audience/stats",
                "weak citation",
                "facebook",
            )
        ):
            dim_scores["content_quality"] = min(
                int(dim_scores.get("content_quality", 50)),
                65,
            )
            dim_scores["seo_compliance"] = min(
                int(dim_scores.get("seo_compliance", 50)),
                70,
            )
            dim_scores["factual_grounding"] = min(
                int(dim_scores.get("factual_grounding", 50)),
                60,
            )

        score = self._calculate_score(dim_scores)
        status = "PASS" if score >= PASS_THRESHOLD else "FAIL"
        needs_revision = status == "FAIL"

        # Always revise when AI tells remain or natural_voice is weak
        if found_tells or int(dim_scores.get("natural_voice", 100)) < 75:
            needs_revision = True
            status = "FAIL"

        rewrite_instruction = ""
        if needs_revision:
            rewrite_instruction = (llm_result.get("rewrite_instruction") or "").strip()
            if not rewrite_instruction:
                rewrite_instruction = self._fallback_rewrite_instruction(
                    dim_scores=dim_scores,
                    issues=pre_check_issues + llm_result.get("issues", []),
                    content_type=str(strategy.get("content_type", "article")).lower(),
                )
            if found_tells:
                rewrite_instruction = (
                    rewrite_instruction
                    + "\nRemove these AI-cliché phrases entirely and rewrite those "
                    "sentences naturally: "
                    + ", ".join(f'"{t}"' for t in found_tells[:8])
                    + ". Never start sentences with Moreover, Furthermore, Additionally, "
                    "or In conclusion. Improve natural_voice above 75."
                ).strip()

        review = {
            "score": score,
            "status": status,
            "needs_revision": needs_revision,
            "feedback": llm_result.get("feedback", []),
            "issues": pre_check_issues + llm_result.get("issues", []),
            "rewrite_instruction": rewrite_instruction,
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

        # Primary / secondary SEO checks (fair matching — not brittle exact-only)
        seo = strategy.get("seo", {}) or {}
        primary_keywords = [
            str(k).strip()
            for k in (
                seo.get("primary_keywords")
                or strategy.get("keywords")
                or []
            )
            if str(k).strip()
        ]
        secondary_keywords = [
            str(k).strip()
            for k in (
                seo.get("secondary_keywords")
                or strategy.get("secondary_keywords")
                or []
            )
            if str(k).strip()
        ]
        draft_lower = draft.lower()

        if primary_keywords:
            lead = primary_keywords[0]
            if not self._keyword_covered(lead, draft_lower):
                issues.append(
                    f"Lead primary keyword not adequately covered in content: {lead}."
                )
            else:
                missing_other = [
                    kw for kw in primary_keywords[1:2]
                    if not self._keyword_covered(kw, draft_lower)
                ]
                if missing_other:
                    issues.append(
                        f"Primary keywords weakly covered: {', '.join(missing_other)}."
                    )

            if content_type in ("blog", "article"):
                h1_match = re.search(r"^#\s+(.+)$", draft, re.MULTILINE)
                if h1_match and not self._keyword_covered(lead, h1_match.group(1).lower()):
                    issues.append(
                        f"Lead primary keyword weakly covered in H1 title: {lead}."
                    )

        # Secondary: only enforce shorter, placeable phrases (≤5 words).
        placeable_secondary = [
            kw for kw in secondary_keywords[:6]
            if len(kw.split()) <= 5
        ]
        if placeable_secondary and content_type in ("blog", "article"):
            hit = any(
                self._keyword_covered(kw, draft_lower)
                for kw in placeable_secondary
            )
            if not hit:
                issues.append(
                    "Secondary keywords weakly covered. "
                    f"Naturally include at least one of: {', '.join(placeable_secondary[:3])}."
                )

        # Soft density band for lead primary (warn only).
        if primary_keywords and content_type in ("blog", "article") and word_count > 0:
            lead = primary_keywords[0].lower()
            escaped = re.escape(lead)
            count = len(re.findall(r"\b" + escaped + r"\b", draft_lower))
            density_pct = (count / word_count) * 100.0
            if count > 0 and density_pct > 3.0:
                issues.append(
                    f"Lead primary keyword may be overused "
                    f"({density_pct:.1f}% density; aim for ~0.5–2.5%)."
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

        # AI-tell phrase check (natural voice)
        found_tells = self._detect_ai_tells(draft_lower)
        if found_tells:
            issues.append(
                "Reads as AI-generated — remove/replace clichéd phrases: "
                + ", ".join(f'"{p}"' for p in found_tells[:6])
                + ". Rewrite in a natural human voice."
            )

        # Sentence-rhythm uniformity (robotic cadence) for long-form
        if content_type in ("blog", "article"):
            uniformity = self._sentence_length_uniformity(self._strip_markdown(draft))
            if uniformity is not None and uniformity < 0.28:
                issues.append(
                    "Sentence rhythm is too uniform (robotic). Vary sentence "
                    "length — mix short punchy sentences with longer ones."
                )

        return issues

    @staticmethod
    def _detect_ai_tells(text_lower: str) -> List[str]:
        """Return AI-cliché phrases present in the draft."""
        found: List[str] = []
        for phrase in AI_TELL_PHRASES:
            if "..." in phrase:
                a, b = [p.strip() for p in phrase.split("...")]
                if re.search(re.escape(a) + r".{0,40}" + re.escape(b), text_lower):
                    found.append(phrase)
            elif phrase in text_lower:
                found.append(phrase)
        return found

    @staticmethod
    def _sentence_length_uniformity(text: str) -> float | None:
        """
        Coefficient of variation of sentence word-counts.
        Low value = uniform/robotic; higher = more human variation.
        Returns None when there are too few sentences to judge.
        """
        sentences = [s.strip() for s in re.split(r"[.!?]+\s+", text) if s.strip()]
        lengths = [len(s.split()) for s in sentences if len(s.split()) > 2]
        if len(lengths) < 6:
            return None
        mean = sum(lengths) / len(lengths)
        if mean == 0:
            return None
        variance = sum((n - mean) ** 2 for n in lengths) / len(lengths)
        std = variance ** 0.5
        return std / mean

    @staticmethod
    def _keyword_covered(keyword: str, text_lower: str) -> bool:
        """
        True if the keyword (or most of its content tokens) appears in text.

        Exact phrase match preferred; otherwise ≥70% of meaningful tokens.
        Avoids failing reviews on near-matches when content is on-topic.
        """
        kw = (keyword or "").strip().lower()
        if not kw:
            return True
        if kw in text_lower:
            return True

        stop = {
            "a", "an", "the", "for", "to", "of", "in", "on", "and", "or",
            "how", "what", "with", "from", "by",
        }
        parts = [
            p for p in re.findall(r"[a-z0-9]+", kw)
            if len(p) > 2 and p not in stop
        ]
        if not parts:
            return False
        hits = sum(1 for p in parts if p in text_lower)
        need = max(1, int((len(parts) * 7 + 9) // 10))  # ceil(0.7 * n)
        return hits >= need

    # ------------------------------------------------------------------
    # LLM evaluation
    # ------------------------------------------------------------------

    def _evaluate_via_llm(
        self,
        draft: str,
        strategy: Dict,
        brand_context: Dict,
        pre_check_issues: List[str],
        primary_topic: str = "",
    ) -> Dict:
        """
        Run a single OpenAI call that scores all six dimensions
        and produces actionable feedback.
        """
        prompt = self._build_evaluation_prompt(
            draft=draft,
            strategy=strategy,
            brand_context=brand_context,
            pre_check_issues=pre_check_issues,
            primary_topic=primary_topic,
        )
        try:
            response = self._openai.chat.completions.create(
                model=self._model,
                max_tokens=1024,
                temperature=self._temperature,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior content editor and SEO strategist. "
                            "You evaluate content objectively and give precise, actionable feedback. "
                            "Return valid JSON only — no prose, no markdown fences."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return self._parse_evaluation(response.choices[0].message.content or "")
        except Exception as exc:
            logger.error("ReviewService LLM call failed: %s — using fallback scores", exc)
            return self._fallback_evaluation(pre_check_issues)

    def _build_evaluation_prompt(
        self,
        draft: str,
        strategy: Dict,
        brand_context: Dict,
        pre_check_issues: List[str],
        primary_topic: str = "",
    ) -> str:
        """Build the structured evaluation prompt."""
        seo = strategy.get("seo", {})
        primary_kw = seo.get("primary_keywords") or strategy.get("keywords", [])
        secondary_kw = seo.get("secondary_keywords") or []
        tone = brand_context.get("tone") or strategy.get("tone", "professional")
        brand_name = (
            brand_context.get("display_name")
            or brand_context.get("brand_name")
            or strategy.get("brand")
            or "the brand"
        )
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

        truncated_draft = self._prepare_draft_for_review(draft)
        secondary_rewrite_hint = (
            "Also fix secondary-keyword gaps in intro/closing and any incomplete sentences. "
            if str(content_type).lower() in ("blog", "article")
            else "Do not keyword-stuff; keep short-form copy natural. Fix any incomplete sentences. "
        )
        brief = (primary_topic or "").strip() or "(not provided)"
        geo_hint = ""
        if re.search(
            r"\b(india|united\s+states|u\.s\.a?\.?|usa|united\s+kingdom|u\.k\.|uk|"
            r"canada|australia|uae|singapore)\b",
            brief,
            re.I,
        ):
            geo_hint = (
                "- If the brief names a geography/market, score content_quality and "
                "factual_grounding DOWN when the draft substitutes unrelated-country "
                "forum or hiring-cost stats instead of sources matching the brief.\n"
            )

        return f"""Evaluate the following {content_type} draft.

IMPORTANT REVIEW RULES:
- Score relative to the USER BRIEF / PRIMARY TOPIC below — not a different subject.
- If the draft changed the topic, omitted requested geography/data, or wrote a generic
  substitute article, content_quality MUST be below 70 and overall should FAIL.
- If the brief asks for audience-specific stats (e.g. NRI) or a year range (e.g. 2022–2025),
  score factual_grounding DOWN when the draft only uses off-audience general scam percentages,
  invents anonymous anecdotes, or cites Facebook posts as primary evidence.
- Scores of 95–100 ONLY when the draft actually fulfills the user brief with only minor polish needed.
- Do NOT inflate scores for a well-written article on the WRONG topic.
- Brand criteria ARE provided below — do NOT claim tone/audience were unspecified.
- If an H2 OUTLINE block is present, do NOT assume middle body sections are missing —
  truncation is for context-window limits only; judge from intro + outline + closing.
- Do NOT flag mid-sentence cutoffs at excerpt boundaries as draft defects when an H2 OUTLINE
  block is present — those cuts are review-window artifacts, not publishing errors.
- Do not invent issues that are not visible in the provided excerpts.
- Prefer specific actionable feedback over harsh generic deductions.
- Score factual_grounding 90+ when the draft uses at least 2–3 clear attributed statistics/citations
  (named source + concrete figure) relevant to the brief and avoids absolute uncited industry claims.
- Score cta_effectiveness 90+ only when the closing CTA matches or closely matches: {cta or "(brand CTA)"}
- Score natural_voice 90+ ONLY when the writing reads like a skilled human wrote it:
  varied sentence length and rhythm, natural transitions, some personality, and NO AI-cliché
  phrases (e.g. "in today's fast-paced world", "moreover", "furthermore", "in conclusion",
  "it's worth noting", "dive in", "game-changer", "unlock the power", "a testament to").
  Deduct heavily for robotic uniform cadence, formulaic scaffolding, or generic filler.
{geo_hint}
=== USER BRIEF / PRIMARY TOPIC (must match) ===
{brief[:500]}

=== EVALUATION CRITERIA ===
BRAND                : {brand_name}
EXPECTED TONE        : {tone}
TARGET AUDIENCE      : {audience_str}
PAIN POINTS TO ADDRESS: {pain_str}
PRIMARY KEYWORDS     : {primary_str}
SECONDARY KEYWORDS   : {secondary_str}
SEARCH INTENT        : {search_intent}
REQUIRED CTA         : {cta or "none"}

=== PRE-CHECK ISSUES (already identified) ===
{pre_issues_str}

=== DRAFT ===
{truncated_draft}

=== SCORING RUBRIC ===
Score each dimension 0–100:

content_quality (weight 18%)
  90–100: Exceptional depth, clear structure, compelling narrative — AND on the user brief
  70–89 : Good coverage of the brief, minor gaps in depth or clarity
  50–69 : Adequate but thin, or partially off-brief
  0–49  : Poor — vague, superficial, or off-topic vs the user brief

seo_compliance (weight 22%)
  90–100: Primary keywords in title, headings, and body; ideal density; keywords match the brief
  70–89 : Keywords mostly present; minor optimisation gaps
  50–69 : Keywords present but not well distributed or weakly related to the brief
  0–49  : Keywords missing from headings or severely underused / wrong-topic keywords

brand_alignment (weight 18%)
  90–100: Tone, audience, and pain points perfectly addressed
  70–89 : Mostly aligned; minor tone or audience mismatch
  50–69 : Some misalignment in tone or audience targeting
  0–49  : Wrong tone, wrong audience, pain points not addressed

structure (weight 12%)
  90–100: Clear intro → body → conclusion, logical flow, good heading hierarchy
  70–89 : Good structure with minor flow issues
  50–69 : Structure present but transitions are weak
  0–49  : Poor structure — missing intro or conclusion, no logical progression

factual_grounding (weight 15%)
  90–100: Claims are supported by research, statistics are attributed, audience/year
          range match the brief; no Facebook-as-primary evidence; no invented anecdotes
  70–89 : Most claims are supported with minor attribution gaps
  50–69 : Some unsupported statements, off-audience stats (e.g. general adult scam %
          when brief asked for NRI), or vague / wrong-geography statistics
  0–49  : Major claims are unsupported, hallucinated, or clearly off-brief on data asks

natural_voice (weight 10%) — HOW HUMAN IT READS
  90–100: Reads like a skilled human writer; varied sentence rhythm, natural flow,
          genuine personality, zero AI-cliché phrases
  70–89 : Mostly natural; a few generic phrases or slightly uniform cadence
  50–69 : Noticeably AI-like — formulaic transitions, repetitive structure, filler
  0–49  : Clearly machine-generated — heavy clichés ("in today's world", "moreover",
          "in conclusion"), robotic uniform sentences, no human voice

cta_effectiveness (weight 5%)
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
    "factual_grounding": <int 0-100>,
    "natural_voice": <int 0-100>,
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
  "rewrite_instruction": "<If weighted score would be < {PASS_THRESHOLD}: one concise paragraph (maximum 150 words) of actionable revision guidance for the Writer Agent. Lead with the lowest-scoring dimension (especially factual_grounding: add 2–3 attributed research stats/citations when appropriate for this content type; and natural_voice: remove AI-cliché phrases, vary sentence rhythm, write in a natural human voice). {secondary_rewrite_hint}If off-brief, rewrite to match the USER BRIEF. If score >= {PASS_THRESHOLD}: empty string.>"
}}
"""

    @staticmethod
    def _topic_fidelity_issue(brief: str, draft: str) -> str:
        """
        Flag when the draft clearly abandoned the user brief (wrong topic / geography).
        Soft heuristic only — Review LLM still judges depth.
        """
        brief_core = (brief or "").split("|")[0].strip().lower()
        draft_l = (draft or "").lower()
        if len(brief_core) < 24:
            return ""

        stop = {
            "write", "article", "about", "the", "and", "for", "with", "from",
            "that", "this", "should", "have", "been", "where", "between",
            "add", "angle", "how", "can", "help", "such", "content", "please",
        }
        tokens = [
            w
            for w in re.findall(r"[a-z0-9]{4,}", brief_core)
            if w not in stop
        ]
        if len(tokens) < 4:
            return ""
        hits = sum(1 for t in tokens[:12] if t in draft_l)
        if hits < max(2, len(tokens[:12]) // 3):
            return (
                "Draft appears off-brief vs the user primary topic — rewrite to match "
                "the requested subject, geography, and data asks (do not substitute a "
                "generic screening essay)."
            )
        if re.search(r"\bindia\b", brief_core) and not re.search(
            r"\b(india|indian|delhi|ncr|gurgaon|mumbai|pocso|ncrb)\b",
            draft_l,
        ):
            if re.search(
                r"\b(usd\s*\d+|reddit|ihss|care\.com|united states|u\.s\.)\b",
                draft_l,
            ):
                return (
                    "Brief requires India geography/data, but draft leans on "
                    "unrelated US/forum sources — use India-relevant evidence."
                )
        # Generic: brief named US/UK but draft is clearly another market dump
        if re.search(r"\b(united\s+states|u\.s\.a?\.?|usa)\b", brief_core) and re.search(
            r"\b(ncrb|pocso|rupees?|₹)\b", draft_l
        ):
            if not re.search(r"\b(united\s+states|u\.s\.|usa|american)\b", draft_l):
                return (
                    "Brief requires US geography/data, but draft leans on "
                    "unrelated India-specific sources — match the brief market."
                )

        # Year-range asks (e.g. 2022 to 2025) must appear or be honestly hedged
        brief_years = set(re.findall(r"\b(20[12]\d)\b", brief_core))
        if len(brief_years) >= 2:
            draft_years = set(re.findall(r"\b(20[12]\d)\b", draft_l))
            if not (brief_years & draft_years):
                return (
                    "Brief asks for a specific year-range of statistics, but the draft "
                    "lacks those years — add on-range attributed figures or explicitly "
                    "state that matching year-range data was unavailable."
                )

        # Audience-specific stats (NRI) must not be replaced only by general-population scam %
        if re.search(r"\bnri\b|non[-\s]?resident", brief_core):
            has_nri = bool(re.search(r"\bnri\b|non[-\s]?resident", draft_l))
            only_general = bool(
                re.search(
                    r"\b(75%\s+of\s+adults|three\s+out\s+of\s+four\s+adults|"
                    r"adults\s+in\s+india\s+have\s+encountered)\b",
                    draft_l,
                )
            )
            if has_nri and only_general and not re.search(
                r"\bnri.{0,40}\b(\d|percent|%|crore|cases)\b|"
                r"\b(\d|percent|%|crore|cases).{0,40}\bnri\b",
                draft_l,
            ):
                return (
                    "Audience/stats mismatch: brief asks for NRI scam statistics, but the "
                    "draft mainly uses general India adult scam percentages — replace with "
                    "NRI/property-specific figures or disclose that NRI-specific series "
                    "were not found."
                )

        # Weak social citations as primary evidence
        if draft_l.count("facebook.com") >= 2:
            return (
                "Weak citation pattern: draft repeatedly cites Facebook posts/videos — "
                "prefer .gov / major news sources for statistics."
            )

        return ""

    @staticmethod
    def _prepare_draft_for_review(draft: str, max_chars: int = 7500) -> str:
        """
        Truncate long drafts without falsely implying middle sections are missing.
        Preserves intro, H2 outline, and closing — cuts at sentence boundaries.
        """
        text = draft or ""
        if len(text) <= max_chars:
            return text

        headings = re.findall(r"^##\s+.+$", text, re.MULTILINE)
        outline = "\n".join(headings[:14]) if headings else "(no H2 headings found)"
        head = ReviewService._cut_at_sentence_boundary(text, 2800, from_end=False)
        tail = ReviewService._cut_at_sentence_boundary(text, 2200, from_end=True)
        return (
            f"{head}\n\n"
            f"=== H2 OUTLINE (full article has these sections) ===\n"
            f"{outline}\n\n"
            f"=== CLOSING ===\n"
            f"{tail}"
        )

    @staticmethod
    def _cut_at_sentence_boundary(text: str, max_chars: int, from_end: bool) -> str:
        """Trim to max_chars without leaving a dangling mid-sentence fragment."""
        if len(text) <= max_chars:
            return text
        if from_end:
            chunk = text[-max_chars:]
            match = re.search(r"(?<=[.!?])\s+", chunk)
            return chunk[match.end():] if match else chunk
        chunk = text[:max_chars]
        matches = list(re.finditer(r"[.!?](?:\s|$)", chunk))
        if matches:
            return chunk[: matches[-1].end()].rstrip()
        return chunk.rstrip()

    def _parse_evaluation(self, raw: str) -> Dict:
        """Parse and validate the LLM evaluation JSON response."""
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        try:
            match = re.search(
                r"\{.*\}",
                cleaned,
                re.DOTALL,
            )

            if not match:
                raise ValueError(
                    "No JSON object found in review response."
                )

            data = json.loads(match.group())  

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
        """Return a conservative evaluation when the LLM call fails — never auto-PASS."""
        base_score = 55 if not pre_check_issues else 45
        return {
            "dimension_scores": {d: base_score for d in DIMENSION_WEIGHTS},
            "feedback": ["Review service fallback — LLM evaluation unavailable."],
            "issues": list(pre_check_issues)
            + ["Automated review unavailable; treat score as provisional."],
            "rewrite_instruction": (
                "Review the draft against the user brief for topic fidelity, "
                "requested geography/data, and brand CTA. Fix any off-brief sections."
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

    @staticmethod
    def _fallback_rewrite_instruction(
        dim_scores: Dict[str, int],
        issues: List[str],
        content_type: str = "article",
    ) -> str:
        """Build rewrite guidance when the LLM left rewrite_instruction empty."""
        lowest = min(DIMENSION_WEIGHTS.keys(), key=lambda d: dim_scores.get(d, 0))
        issue_bits = "; ".join(str(i) for i in issues[:3] if str(i).strip())
        secondary_bit = (
            "place secondary keywords naturally in the introduction and conclusion; "
            if str(content_type).lower() in ("blog", "article")
            else "do not keyword-stuff; "
        )
        base = (
            f"Revise to reach an overall score of at least {PASS_THRESHOLD}. "
            f"Priority dimension: {lowest}. "
            "Add attributed statistics from research only when available (named source + "
            "concrete figure/percentage/year — never invent orgs or vague 'studies show'); "
            "remove absolute uncited industry claims; "
            f"{secondary_bit}"
            "complete every sentence; "
            "close with the brand CTA verbatim (specific action, not 'reach out today'); "
            "write currency as USD amounts without the $ character; "
            "rewrite in a natural human voice — ban Moreover/Furthermore/In conclusion/"
            "it's worth noting; vary sentence length; keep natural_voice above 75."
        )
        if issue_bits:
            return f"{base} Also address: {issue_bits}"
        return base

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
