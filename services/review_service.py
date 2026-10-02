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
  Content Quality    18 % — depth, clarity, demonstration, value
  SEO Compliance     14 % — natural keyword presence (not stuffing)
  Brand Alignment    18 % — tone match, audience fit, pain points addressed
  Structure          12 % — intro / body / conclusion, heading hierarchy
  Factual Grounding  15 % — attributed, on-brief evidence
  Natural Voice      18 % — human cadence, no template / B2B cliché
  CTA Effectiveness   5 % — clear, action-oriented, intent-matched

PASS threshold : score >= 95
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from config.settings import settings

logger = logging.getLogger(__name__)

PASS_THRESHOLD = 95

DIMENSION_WEIGHTS: Dict[str, float] = {
    "content_quality": 0.18,
    "seo_compliance": 0.14,
    "brand_alignment": 0.18,
    "structure": 0.12,
    "factual_grounding": 0.15,
    "natural_voice": 0.18,
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
    "this article explores",
    "first, discuss",
    "first discuss",
    "let us examine",
    "as we delve",
    "expatriates",
    "expatriate",
    # Cinematic / brochure tells (generic — all brands)
    "picture a family",
    "picture this",
    "imagine a weekday",
    "imagine a family",
    "the stakes are high",
    "your family deserves nothing less",
    "the stakes are high",
    "cutting-edge",
    "holistic approach",
    "seamless experience",
    "delve into",
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

        # Review scoring uses the light model by default (cost).
        # Pre-checks still catch length/keyword/topic issues without the LLM.
        # Override with OPENAI_MODEL_REVIEW=gpt-4.1 if scoring quality dips.
        self._model = settings.model_for_review()
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

        # Mode policy review weights (Content Quality OS)
        self._active_weights = dict(DIMENSION_WEIGHTS)
        try:
            from config.mode_policies import get_policy, review_weights_for

            pol = {}
            if isinstance(brand_context, dict):
                pol = brand_context.get("mode_policy") or {}
                if not pol and brand_context.get("content_mode"):
                    pol = get_policy(str(brand_context.get("content_mode")))
            if pol:
                self._active_weights = review_weights_for(pol)
        except Exception:
            self._active_weights = dict(DIMENSION_WEIGHTS)

        # Hard fail empty drafts immediately (no LLM spend on blank content).
        if not (draft or "").strip():
            content_type = str(strategy.get("content_type", "article")).lower()
            secondary_bit = (
                " Prefer natural wording over keyword stuffing."
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

        content_type_early = str(strategy.get("content_type") or "").lower()
        platform_early = str(strategy.get("platform") or "").lower()
        if content_type_early == "comment" or platform_early == "comment":
            return self._evaluate_comment(
                draft=draft,
                strategy=strategy,
                revision_count=revision_count,
                primary_topic=brief,
            )

        # Rule-based pre-checks (fast, no LLM)
        pre_check_issues = self._run_pre_checks(
            draft, strategy, brand_context=brand_context, primary_topic=brief
        )
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
        # Cap natural_voice when AI-cliché / humanize pre-check fired
        found_tells = self._detect_ai_tells(draft.lower())
        humanize_flags = [
            i for i in pre_check_issues if str(i).startswith("HUMANIZE_")
        ]
        if found_tells or humanize_flags:
            dim_scores["natural_voice"] = min(
                int(dim_scores.get("natural_voice", 50)),
                55 if humanize_flags else 60,
            )

        # Soft vs severe fidelity: do not crush scores for honest data gaps.
        fidelity_flags = " ".join(str(i).lower() for i in pre_check_issues)
        severe = any(
            flag in fidelity_flags
            for flag in (
                "off-brief",
                "india geography",
                "us geography",
                "weak citation",
                "facebook",
                "audience/stats mismatch",
            )
        )
        soft = any(
            flag in fidelity_flags
            for flag in ("year-range", "matching year-range data")
        )
        if severe:
            # Real topic/geo/citation failures — keep caps, but not as harsh as before
            dim_scores["content_quality"] = min(
                int(dim_scores.get("content_quality", 50)),
                78,
            )
            dim_scores["seo_compliance"] = min(
                int(dim_scores.get("seo_compliance", 50)),
                80,
            )
            dim_scores["factual_grounding"] = min(
                int(dim_scores.get("factual_grounding", 50)),
                72,
            )
        elif soft:
            # Missing years with no honest hedge — nudge grounding only
            dim_scores["factual_grounding"] = min(
                int(dim_scores.get("factual_grounding", 50)),
                82,
            )

        score = self._calculate_score(dim_scores)
        status = "PASS" if score >= PASS_THRESHOLD else "FAIL"
        needs_revision = status == "FAIL"

        # Always revise when AI tells remain, humanize QC failed, or natural_voice is weak
        if (
            found_tells
            or humanize_flags
            or int(dim_scores.get("natural_voice", 100)) < 75
        ):
            needs_revision = True
            status = "FAIL"

        # Publishing QC gates — force a rewrite when these fire
        qc_force = [
            i
            for i in pre_check_issues
            if any(
                key in str(i)
                for key in (
                    "TEMPORAL_MISMATCH",
                    "BRAND_CTA_MISMATCH",
                    "DUPLICATE_HASHTAGS",
                    "ABSOLUTE_CLAIM",
                    "HEADING_CLAIM_MISMATCH",
                    "HUMANIZE_",
                    "DEMONSTRATE_",
                    "FIDELITY_OPENER",
                    "FIDELITY_BRAND_NAME",
                    "FIDELITY_MARKET",
                    "EVIDENCE_UNSUPPORTED",
                    "EVIDENCE_OVERCLAIM",
                    "EVIDENCE_TONE",
                )
            )
        ]
        if qc_force:
            needs_revision = True
            status = "FAIL"
            dim_scores["factual_grounding"] = min(
                int(dim_scores.get("factual_grounding", 50)),
                70 if any("EVIDENCE_" in str(i) for i in qc_force) else 78,
            )
            dim_scores["brand_alignment"] = min(
                int(dim_scores.get("brand_alignment", 50)),
                82,
            )
            if any(str(i).startswith("HUMANIZE_") for i in qc_force):
                dim_scores["natural_voice"] = min(
                    int(dim_scores.get("natural_voice", 50)),
                    50,
                )
            if any(str(i).startswith("DEMONSTRATE_") for i in qc_force):
                # ~8.5 ceiling until abstract claims are shown in real business scenes
                dim_scores["content_quality"] = min(
                    int(dim_scores.get("content_quality", 50)),
                    84,
                )
            score = self._calculate_score(dim_scores)

        rewrite_instruction = ""
        if needs_revision:
            rewrite_instruction = (llm_result.get("rewrite_instruction") or "").strip()
            if not rewrite_instruction:
                rewrite_instruction = self._fallback_rewrite_instruction(
                    dim_scores=dim_scores,
                    issues=pre_check_issues + llm_result.get("issues", []),
                    content_type=str(strategy.get("content_type", "article")).lower(),
                )
            demo_flags = [
                i for i in pre_check_issues if str(i).startswith("DEMONSTRATE_")
            ]
            if demo_flags:
                rewrite_instruction = (
                    (rewrite_instruction + "\n" if rewrite_instruction else "")
                    + "9+ RULE — demonstrate, do not only explain: for every abstract "
                    "claim, show what it looks like in a real business (workflow moment, "
                    "decision trade-off, cost/time impact, or team scene). Keep the "
                    "ideas; replace explanatory passages with concrete scenes. "
                    + " ".join(str(d) for d in demo_flags[:2])
                ).strip()
            if found_tells:
                rewrite_instruction = (
                    rewrite_instruction
                    + "\nRemove these AI-cliché phrases entirely and rewrite those "
                    "sentences naturally: "
                    + ", ".join(f'"{t}"' for t in found_tells[:8])
                    + ". Never start sentences with Moreover, Furthermore, Additionally, "
                    "or In conclusion. Improve natural_voice above 75."
                ).strip()
            if qc_force:
                rewrite_instruction = (
                    rewrite_instruction
                    + "\nPUBLISHING QC FIXES (mandatory):\n- "
                    + "\n- ".join(qc_force[:8])
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

    def _run_pre_checks(
        self,
        draft: str,
        strategy: Dict,
        brand_context: Optional[Dict] = None,
        primary_topic: str = "",
    ) -> List[str]:
        """
        Fast, rule-based checks that run before the LLM call.
        Returns a list of issue strings (empty = all passed).
        """
        issues: List[str] = []

        word_count = len(self._strip_markdown(draft).split())
        content_type = strategy.get("content_type", "article")

        # Word count: user target wins over default article floors.
        user_target = strategy.get("target_word_count")
        try:
            user_target_n = int(user_target) if user_target is not None else None
        except (TypeError, ValueError):
            user_target_n = None

        if user_target_n and 1 <= user_target_n <= 50000:
            flexible = bool(strategy.get("word_count_flexible", True))
            if user_target_n <= 50:
                lo, hi = max(1, user_target_n - 2), user_target_n + 2
            elif flexible:
                lo, hi = int(user_target_n * 0.85), int(user_target_n * 1.15)
            else:
                lo, hi = int(user_target_n * 0.95), int(user_target_n * 1.05)
            if word_count < lo or word_count > hi:
                issues.append(
                    f"Content length {word_count} words is outside the "
                    f"user-requested ~{user_target_n} words (acceptable {lo}-{hi})."
                )
        elif content_type in ("blog", "article"):
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

        # Micro length asks: skip long-form SEO/heading rules (they force expansion).
        micro_request = bool(user_target_n and user_target_n <= 75)

        from config.mode_policies import (
            brand_allowed_in_h1,
            get_policy,
            is_awareness_mode,
            secondaries_enforced,
        )
        from services.cta_policy import is_awareness_first

        policy = {}
        if isinstance(brand_context, dict):
            policy = brand_context.get("mode_policy") or {}
            if not policy:
                mode = str(brand_context.get("content_mode") or "")
                if mode:
                    policy = get_policy(mode)
        awareness = is_awareness_mode(policy) or (
            not policy and is_awareness_first(brand_context)
        )
        topic_h1 = (not brand_allowed_in_h1(policy)) if policy else awareness
        late_brand = str(policy.get("brand_placement") or "") == "late_third" or (
            awareness and not policy
        )
        enforce_secondaries = (
            secondaries_enforced(policy) if policy else (not awareness)
        )
        # Primary / secondary SEO checks (fair matching — not brittle exact-only)
        seo = strategy.get("seo", {}) or {}
        brand_keywords = [
            str(k).strip()
            for k in (
                seo.get("brand_keywords")
                or ((brand_context or {}).get("seo_kit_selected") or {}).get(
                    "brand_keywords"
                )
                or []
            )
            if str(k).strip()
        ]
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
        brand_keys_l = {b.lower() for b in brand_keywords}
        # If kit brand terms were merged into primary, treat them as brand not topic lead
        if not brand_keys_l:
            # Heuristic: CamelCase / Care / brand display name
            display = str((brand_context or {}).get("display_name") or "").lower()
            for p in primary_keywords:
                pl = p.lower().replace(" ", "")
                if display and display.replace(" ", "") in pl:
                    brand_keys_l.add(p.lower())
                    brand_keywords.append(p)
        topic_primaries = [
            p for p in primary_keywords if p.lower() not in brand_keys_l
        ]

        if primary_keywords and not micro_request:
            if topic_h1:
                lead = topic_primaries[0] if topic_primaries else ""
                if lead and not self._keyword_covered(lead, draft_lower):
                    issues.append(
                        f"Lead topic keyword not adequately covered in content: {lead}."
                    )
                if content_type in ("blog", "article") and lead:
                    h1_match = re.search(r"^#\s+(.+)$", draft, re.MULTILINE)
                    if h1_match and not self._keyword_covered(
                        lead, h1_match.group(1).lower()
                    ):
                        issues.append(
                            f"Lead topic keyword weakly covered in H1 title: {lead}."
                        )
                for bk in brand_keywords[:1]:
                    if not self._keyword_covered(bk, draft_lower):
                        if late_brand or awareness:
                            issues.append(
                                f"Brand keyword missing (late-placement): {bk}. "
                                "Mention it once near the end / CTA — do not put it in the H1."
                            )
                    else:
                        h1_match = re.search(r"^#\s+(.+)$", draft, re.MULTILINE)
                        if (
                            h1_match
                            and not brand_allowed_in_h1(policy)
                            and self._keyword_covered(bk, h1_match.group(1).lower())
                        ):
                            issues.append(
                                f"MODE_SEO: Brand keyword \"{bk}\" should not lead the H1 "
                                f"in {policy.get('mode') or 'this'} mode. "
                                "Use a topic phrase in the title; place the brand later."
                            )
                        if late_brand:
                            mid = max(1, len(draft_lower) // 2)
                            if not self._keyword_covered(bk, draft_lower[mid:]):
                                issues.append(
                                    f"Brand keyword \"{bk}\" appears too early. "
                                    "Keep awareness pacing — introduce the brand in the final third."
                                )
            else:
                lead = primary_keywords[0]
                if not self._keyword_covered(lead, draft_lower):
                    issues.append(
                        f"Lead primary keyword not adequately covered in content: {lead}."
                    )
                else:
                    missing_other = [
                        kw
                        for kw in primary_keywords[1:2]
                        if not self._keyword_covered(kw, draft_lower)
                    ]
                    if missing_other:
                        issues.append(
                            f"Primary keywords weakly covered: {', '.join(missing_other)}."
                        )

                if content_type in ("blog", "article"):
                    h1_match = re.search(r"^#\s+(.+)$", draft, re.MULTILINE)
                    if h1_match and not self._keyword_covered(
                        lead, h1_match.group(1).lower()
                    ):
                        issues.append(
                            f"Lead primary keyword weakly covered in H1 title: {lead}."
                        )

        # Secondary: only enforce shorter, placeable phrases (≤5 words).
        # Optional modes (awareness/authority): do not force secondaries.
        placeable_secondary = [
            kw
            for kw in secondary_keywords[:6]
            if len(kw.split()) <= 5 and kw.lower() not in brand_keys_l
        ]
        if (
            placeable_secondary
            and content_type in ("blog", "article")
            and not micro_request
            and enforce_secondaries
        ):
            hit = any(
                self._keyword_covered(kw, draft_lower)
                for kw in placeable_secondary
            )
            if not hit:
                issues.append(
                    "Secondary keywords weakly covered. "
                    f"Naturally include at least one of: {', '.join(placeable_secondary[:3])}."
                )

        # Soft density band for lead topic/primary (warn only).
        density_lead = (
            (topic_primaries[0] if topic_primaries else "")
            if topic_h1
            else (primary_keywords[0] if primary_keywords else "")
        )
        if (
            density_lead
            and content_type in ("blog", "article")
            and word_count > 0
            and not micro_request
        ):
            lead = density_lead.lower()
            escaped = re.escape(lead)
            count = len(re.findall(r"\b" + escaped + r"\b", draft_lower))
            density_pct = (count / word_count) * 100.0
            if count > 0 and density_pct > 3.0:
                issues.append(
                    f"Lead primary keyword may be overused "
                    f"({density_pct:.1f}% density; aim for ~0.5–2.5%)."
                )

        # Heading structure check (skip for micro asks)
        h2_count = len(re.findall(r"^##\s+", draft, re.MULTILINE))
        if content_type in ("blog", "article") and h2_count < 2 and not micro_request:
            issues.append(
                f"Insufficient headings: found {h2_count} H2 headings (minimum 2)."
            )

        # CTA check (skip for micro — full CTA often won't fit the budget)
        # Soft vs hard policy is shared with Writer (services.cta_policy) so
        # awareness / explain briefs are not force-failed for missing CTA.
        if not micro_request:
            from services.cta_policy import cta_is_hard_required

            brand = brand_context or {}
            cta = str(
                strategy.get("cta")
                or brand.get("cta")
                or brand_context_from_strategy(strategy)
                or ""
            ).strip()
            display_name = str(brand.get("display_name") or "").strip()
            hard_cta = cta_is_hard_required(
                content_type=content_type,
                primary_topic=primary_topic or "",
                objective=str(strategy.get("objective") or brand.get("objective") or ""),
                brand_context=brand,
            )
            if hard_cta and cta and cta.lower() not in draft_lower:
                # Accept brand-name variants of the CTA (e.g. "Contact MPM…")
                brand_cta_ok = bool(
                    display_name
                    and display_name.lower() in draft_lower
                    and re.search(
                        r"\b(contact|book|schedule|call|reach)\b",
                        draft_lower[-900:],
                    )
                )
                if not brand_cta_ok:
                    issues.append(
                        f"BRAND_CTA_MISMATCH: CTA text not found near the close. "
                        f"End with the brand CTA verbatim: \"{cta}\"."
                    )
            elif hard_cta and display_name and content_type in ("blog", "article"):
                # Closing CTA must name the brand, not a generic advisor label
                closing = draft[-1200:]
                generic = re.search(
                    r"contact\s+(a\s+)?(trusted\s+)?property\s+advisor\b",
                    closing,
                    re.I,
                )
                if generic and display_name.lower() not in closing.lower():
                    issues.append(
                        f"BRAND_CTA_MISMATCH: Closing CTA is generic "
                        f"(\"Contact Property Advisor\") but brand is "
                        f"\"{display_name}\". Use the brand name in the final CTA."
                    )

        # AI-tell phrase check (natural voice)
        found_tells = self._detect_ai_tells(draft_lower)
        if found_tells and not micro_request:
            issues.append(
                "Reads as AI-generated — remove/replace clichéd phrases: "
                + ", ".join(f'"{p}"' for p in found_tells[:6])
                + ". Rewrite in a natural human voice."
            )

        # Sentence-rhythm uniformity (robotic cadence) for long-form
        if content_type in ("blog", "article") and not micro_request:
            uniformity = self._sentence_length_uniformity(self._strip_markdown(draft))
            if uniformity is not None and uniformity < 0.28:
                issues.append(
                    "Sentence rhythm is too uniform (robotic). Vary sentence "
                    "length — mix short punchy sentences with longer ones."
                )

        # ---- Publishing QC (temporal, absolute claims, duplicates, headings) ----
        if content_type in ("blog", "article") and not micro_request:
            issues.extend(
                self._publishing_qc_issues(
                    draft=draft,
                    primary_topic=primary_topic or "",
                )
            )

        # ---- Humanize QC (generic — all brands): does it read human? ----
        if content_type in ("blog", "article", "linkedin", "email") and not micro_request:
            try:
                from services.fidelity_gate import humanize_review_issues

                issues.extend(humanize_review_issues(draft))
            except Exception:
                pass

        # ---- 9+ demonstrate QC (articles): explain vs show in a real scene ----
        if content_type in ("blog", "article") and not micro_request:
            try:
                from config.mode_policies import demonstrate_style, get_policy
                from services.fidelity_gate import demonstrate_review_issues

                pol = {}
                if isinstance(brand_context, dict):
                    pol = brand_context.get("mode_policy") or {}
                    if not pol and brand_context.get("content_mode"):
                        pol = get_policy(str(brand_context.get("content_mode")))
                issues.extend(
                    demonstrate_review_issues(
                        draft,
                        content_type=content_type,
                        demo_style=demonstrate_style(pol) if pol else "",
                    )
                )
            except Exception:
                pass

        # ---- Evidence claim audit (generic — all brands) ----
        if content_type in ("blog", "article") and not micro_request:
            try:
                from services.evidence_ledger import claim_audit_issues

                ledger = strategy.get("evidence_ledger") or []
                issues.extend(claim_audit_issues(draft, ledger))
            except Exception:
                pass

        # ---- Generic fidelity gate (all brands) ----
        try:
            from services.fidelity_gate import BriefLock, review_fidelity_issues

            lock = BriefLock.from_dict(
                (brand_context or {}).get("brief_lock")
                or strategy.get("brief_lock")
                or {}
            )
            if not lock.topic:
                lock.topic = primary_topic or ""
            if not lock.brand_display_name:
                lock.brand_display_name = str(
                    (brand_context or {}).get("display_name") or ""
                )
            issues.extend(
                review_fidelity_issues(
                    draft,
                    lock,
                    citations=strategy.get("citations") or [],
                )
            )
        except Exception:
            pass

        return issues

    def _publishing_qc_issues(self, draft: str, primary_topic: str = "") -> List[str]:
        """Rule-based publishing QC: temporal, absolute claims, hashtags, headings."""
        issues: List[str] = []
        text = draft or ""

        # 1) Duplicate hashtag footers
        hashtag_blocks = re.findall(r"(?im)^Hashtags:\s*.+$", text)
        if len(hashtag_blocks) >= 2:
            issues.append(
                "DUPLICATE_HASHTAGS: Hashtag footer appears more than once. "
                "Keep a single Hashtags: line at the end."
            )

        # 2) Temporal mismatch — article year vs later source years
        article_years = set()
        h1 = re.search(r"^#\s+(.+)$", text, re.M)
        title_blob = " ".join(
            p for p in [primary_topic, h1.group(1) if h1 else ""] if p
        )
        for y in re.findall(r"\b(20[2-3]\d)\b", title_blob):
            article_years.add(int(y))
        # Also treat "… Guide 2026" patterns in first H1 as article year
        if h1:
            for y in re.findall(r"\b(20[2-3]\d)\b", h1.group(1)):
                article_years.add(int(y))

        if article_years:
            article_year = min(article_years)
            # Sources cited as "(2027)" or "Infra (2027)" ahead of article year
            future_hits = []
            for m in re.finditer(
                r"([A-Za-z][A-Za-z0-9 .&/-]{2,60})\((20[2-3]\d)\)",
                text,
            ):
                src_year = int(m.group(2))
                if src_year > article_year:
                    future_hits.append(f"{m.group(1).strip()} ({src_year})")
            if future_hits:
                issues.append(
                    "TEMPORAL_MISMATCH: Article is framed for "
                    f"{article_year} but cites later-dated sources "
                    f"({'; '.join(future_hits[:4])}). Remove/replace those "
                    f"citations or reframe the article year to match evidence."
                )

        # 3) Absolute / over-promising marketing language
        absolute_patterns = [
            (r"\brisk[-\s]?free\b", "risk-free"),
            (r"\bguaranteed?\b", "guaranteed"),
            (r"\bunmatched\b", "unmatched"),
            (r"\bmost secure\b", "most secure"),
            (r"\bmore favorable than ever\b", "more favorable than ever"),
            (r"\bcut down the risk of\b", "cut down the risk of"),
            (r"\bwill (?:increase|deliver|guarantee)\b", "will increase/deliver"),
            (r"\bbest (?:investment|choice|opportunity) (?:ever|in the market)\b", "best … ever"),
        ]
        abs_hits = []
        for pat, label in absolute_patterns:
            if re.search(pat, text, re.I):
                abs_hits.append(label)
        if abs_hits:
            issues.append(
                "ABSOLUTE_CLAIM: Soften over-absolute language "
                f"({', '.join(abs_hits[:5])}). Qualify with evidence "
                "(e.g. 'greater oversight… although project-specific risks remain')."
            )

        # 4) Heading promises "predictions" without predictive content
        for hm in re.finditer(r"^##\s+(.+)$", text, re.M):
            heading = hm.group(1).strip()
            if re.search(r"\bpredictions?\b", heading, re.I):
                # Look ahead until next H2
                start = hm.end()
                nxt = re.search(r"^##\s+", text[start:], re.M)
                body = text[start : start + (nxt.start() if nxt else 1200)]
                has_forward = bool(
                    re.search(
                        r"\b(forecast|projected|expected to|by 20\d\d|outlook|"
                        r"price prediction|will reach|estimated)\b",
                        body,
                        re.I,
                    )
                )
                if not has_forward:
                    issues.append(
                        "HEADING_CLAIM_MISMATCH: Heading promises "
                        f"\"{heading}\" but the section lacks forward-looking "
                        "estimates. Rename to 'Price Trends' or add credible "
                        "forecasts with attribution."
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
        Also accepts compound brand forms (KinvoCare ↔ kinvo care).
        """
        kw = (keyword or "").strip().lower()
        if not kw:
            return True
        if kw in text_lower:
            return True
        # CamelCase / spaced brand variants
        spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", keyword or "").strip().lower()
        compact = re.sub(r"\s+", "", kw)
        text_compact = re.sub(r"\s+", "", text_lower)
        if spaced and spaced != kw and spaced in text_lower:
            return True
        if compact and len(compact) >= 4 and compact in text_compact:
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

        from services.cta_policy import cta_is_hard_required, review_cta_instruction

        hard_cta = cta_is_hard_required(
            content_type=str(content_type or ""),
            primary_topic=primary_topic or "",
            objective=str(strategy.get("objective") or brand_context.get("objective") or ""),
            brand_context=brand_context,
        )
        cta_policy_line = review_cta_instruction(str(cta or ""), hard_cta)
        cta_score_hint = (
            f"Score cta_effectiveness 90+ only when the closing CTA matches or closely matches: "
            f"{cta or '(brand CTA)'}"
            if hard_cta
            else (
                "Score cta_effectiveness 85+ when the close is natural and on-brief; "
                "do NOT heavily punish a soft/educational close that omits a hard sell. "
                f"If a CTA appears, prefer: {cta or '(brand CTA)'}."
            )
        )

        audience_str = ", ".join(str(a) for a in audience) if isinstance(audience, list) else str(audience)
        pain_str = "; ".join(str(p) for p in pain_points[:4]) if pain_points else "none"
        primary_str = ", ".join(primary_kw[:5]) if primary_kw else "none"
        secondary_str = ", ".join(secondary_kw[:5]) if secondary_kw else "none"
        pre_issues_str = "\n".join(f"- {i}" for i in pre_check_issues) if pre_check_issues else "None"

        truncated_draft = self._prepare_draft_for_review(draft)
        secondary_rewrite_hint = (
            "Remove keyword stuffing; do not force secondary keywords into intro/closing. "
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
- For awareness-first brands (warm/family educate-before-promote): do NOT demand the
  brand product name in the H1. Score seo_compliance on topic keywords; brand name
  once late near the CTA is correct. Deduct if brand is stuffed early or missing entirely.
- Score factual_grounding 90+ when the draft uses at least 2–3 clear attributed statistics/citations
  (named source + concrete figure) relevant to the brief and avoids absolute uncited industry claims.
- {cta_score_hint}
- Score natural_voice 90+ ONLY when the writing reads like a skilled human wrote it:
  varied sentence length and rhythm, natural transitions, some personality,
  concrete examples, full word forms (no you're / it's / I'd contractions),
  no cinematic Picture/Imagine openers, no brochure closers, and NO AI-cliché
  phrases (e.g. "in today's fast-paced world", "moreover", "furthermore",
  "in conclusion", "it's worth noting", "dive in", "game-changer",
  "unlock the power", "a testament to", "leverage", "cutting-edge",
  "in the landscape/realm of", "holistic", "seamless", "the stakes are high").
  Deduct heavily for robotic uniform cadence, equal-sized template sections,
  brochure phrasing, formulaic scaffolding, or generic filler.
  If PRE-CHECK lists HUMANIZE_* issues, natural_voice MUST be below 60 and
  rewrite_instruction MUST tell the Writer exactly how to humanize those lines.
- Demonstrate > explain (9+ bar for content_quality):
  Cap content_quality at 84 when the draft has good ideas but mostly *explains*
  abstract claims instead of *showing* them in a real business (workflow moment,
  decision trade-off, cost/time impact, team scene, before/after).
  Score content_quality 90+ ONLY when abstract claims are regularly demonstrated
  that way. On FAIL / revision, rewrite_instruction MUST name 2–3 abstract claims
  and require a concrete business demonstration for each.
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
{cta_policy_line}

=== PRE-CHECK ISSUES (already identified) ===
{pre_issues_str}

=== DRAFT ===
{truncated_draft}

=== SCORING RUBRIC ===
Score each dimension 0–100:

content_quality (weight 18%)
  90–100: Strong ideas *demonstrated* in real-business scenes (workflow, decision,
          cost/time, team moment) — not only explained; on the user brief
  70–89 : Good ideas/coverage, but some claims stay abstract/explanatory
          (typical ~8.5 / mid-80s ceiling when demonstration is missing)
  50–69 : Adequate but thin, or partially off-brief
  0–49  : Poor — vague, superficial, or off-topic vs the user brief

seo_compliance (weight 14%)
  90–100: Primary terms appear naturally where useful; no stuffing; keywords match the brief
  70–89 : Keywords mostly present; minor gaps OK if voice stays natural
  50–69 : Awkward / repeated keyword injection or weakly related terms
  0–49  : Severe stuffing OR keywords missing when the brief clearly needs them

brand_alignment (weight 18%)
  90–100: Tone, audience, and pain points perfectly addressed
  70–89 : Mostly aligned; minor tone or audience mismatch
  50–69 : Some misalignment in tone or audience targeting
  0–49  : Wrong tone, wrong audience, pain points not addressed

structure (weight 12%)
  90–100: Clear intro → body → conclusion, logical flow, varied section shapes
  70–89 : Good structure with minor flow issues
  50–69 : Structure present but transitions are weak or sections feel cloned
  0–49  : Poor structure — missing intro or conclusion, no logical progression

factual_grounding (weight 15%)
  90–100: Claims are supported by research, statistics are attributed, audience/year
          range match the brief; no Facebook-as-primary evidence; no invented anecdotes;
          no ornamental off-angle stats added only for authority
  70–89 : Most claims are supported with minor attribution gaps
  50–69 : Some unsupported statements, off-audience stats, ornamental authority numbers,
          or vague / wrong-geography statistics
  0–49  : Major claims are unsupported, hallucinated, or clearly off-brief on data asks

natural_voice (weight 18%) — HOW HUMAN IT READS
  90–100: Reads like a skilled human writer; varied sentence rhythm, natural flow,
          genuine personality, zero AI/B2B-cliché phrases, no thesis restated every H2
  70–89 : Mostly natural; a few generic phrases or slightly uniform cadence
  50–69 : Noticeably AI-like — formulaic transitions, repetitive structure, filler
  0–49  : Clearly machine-generated — heavy clichés ("in today's world", "moreover",
          "in conclusion"), robotic uniform sentences, no human voice

cta_effectiveness (weight 5%)
  90–100: Clear, specific CTA used once, aligned with search intent
  70–89 : CTA present but could be stronger, more specific, or appears more than once
  50–69 : Weak, vague, or duplicated CTA
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
  "rewrite_instruction": "<If weighted score would be < {PASS_THRESHOLD}: one concise paragraph (maximum 150 words) of actionable revision guidance for the Writer Agent. Lead with the lowest-scoring dimension. Prefer SUBTRACTION: remove keyword stuffing, delete repeated thesis lines, drop ornamental stats, keep at most one CTA, kill soft B2B clichés (future-ready / changes the game / the payoff is clear). Only add a research-backed figure if a section truly lacks needed proof — never pad with 2–3 stats for authority. natural_voice: remove AI-cliché phrases, vary sentence rhythm. content_quality: demonstrate remaining abstract claims in a real business scene. {secondary_rewrite_hint}If off-brief, rewrite to match the USER BRIEF. If score >= {PASS_THRESHOLD}: empty string.>"
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
        # Short briefs still get a fidelity check (e.g. "AI automation blog")
        if len(brief_core) < 10:
            return ""

        stop = {
            "write", "article", "about", "the", "and", "for", "with", "from",
            "that", "this", "should", "have", "been", "where", "between",
            "add", "angle", "how", "can", "help", "such", "content", "please",
            "blog", "post", "make", "create", "generate", "want", "need",
            "likho", "bahut", "accha", "jo", "yeh", "very", "good",
        }
        tokens = [
            w
            for w in re.findall(r"[a-z0-9]{3,}", brief_core)
            if w not in stop
        ]
        if len(tokens) < 2:
            return ""
        sample = tokens[:12]
        hits = sum(1 for t in sample if t in draft_l)
        # Require roughly 1/3 of brief tokens (min 1 for micro briefs)
        need = max(1, len(sample) // 3)
        if hits < need:
            return (
                "Draft appears off-brief vs the user primary topic — rewrite to match "
                "the requested subject, geography, and data asks (do not substitute a "
                "generic brand pitch or screening essay)."
            )
        # Brand-pitch hijack: brief is not about sales but draft is mostly CTA/leads
        pitchy = sum(
            1
            for p in (
                "book a",
                "discovery call",
                "lead leakage",
                "schedule a demo",
                "measurable roi",
                "our approach could",
            )
            if p in draft_l
        )
        if pitchy >= 2 and not any(
            p in brief_core for p in ("cta", "call", "demo", "sales", "lead")
        ):
            return (
                "Draft drifted into a brand sales pitch instead of answering the user "
                "brief — rewrite to the requested subject first; keep CTA soft/optional."
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

        # Year-range asks (e.g. 2022 to 2025) must appear OR be honestly hedged
        brief_years = set(re.findall(r"\b(20[12]\d)\b", brief_core))
        if len(brief_years) >= 2:
            draft_years = set(re.findall(r"\b(20[12]\d)\b", draft_l))
            honest_gap = bool(
                re.search(
                    r"\b("
                    r"no publicly available|not separately tracked|data is limited|"
                    r"limited data|nanny-specific (data|statistics) (is|are) (limited|not)|"
                    r"unavailable|harder to isolate|does not always isolate|"
                    r"exact (year|year-?range|figures?).{0,40}(unavailable|not found|limited)"
                    r")\b",
                    draft_l,
                )
            )
            if not (brief_years & draft_years) and not honest_gap:
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

    def _evaluate_comment(
        self,
        draft: str,
        strategy: Dict,
        revision_count: int,
        primary_topic: str = "",
    ) -> Dict:
        """
        Lightweight review for social comments/replies.
        Does NOT require SEO keywords, research stats, brand CTA, or article structure.
        """
        text = (draft or "").strip()
        words = text.split()
        word_count = len(words)
        issues: List[str] = []
        feedback: List[str] = []

        target = strategy.get("target_word_count")
        try:
            target_n = int(target) if target is not None else None
        except (TypeError, ValueError):
            target_n = None

        if target_n and target_n <= 50:
            lo, hi = max(1, target_n - 2), target_n + 2
            if word_count < lo or word_count > hi:
                issues.append(
                    f"Comment length {word_count} words is outside the requested "
                    f"~{target_n} words (acceptable {lo}-{hi})."
                )
        elif word_count > 120:
            issues.append(
                f"Comment is too long ({word_count} words). Keep replies short."
            )

        if re.search(r"(?:#\w+\s*){2,}|^Hashtags:", text, re.I | re.M):
            issues.append("Comments must not include hashtags.")

        # Sales / outreach pitch is off-brief for a thank-you comment
        pitch_hits = 0
        for phrase in (
            "book a",
            "discovery call",
            "schedule a",
            "lead leakage",
            "measurable roi",
            "our approach",
            "i would love to hear",
        ):
            if phrase in text.lower():
                pitch_hits += 1
        if pitch_hits >= 2:
            issues.append(
                "Draft reads like outreach/sales, not a natural social comment reply."
            )

        last_word = re.sub(r"[^\w']+$", "", words[-1]).lower() if words else ""
        dangling = {
            "and", "or", "to", "for", "the", "a", "an", "of", "with", "our",
            "if", "in", "on", "at", "by", "from", "as", "than", "that", "this",
        }
        if text.rstrip().endswith(",") or last_word in dangling:
            issues.append("Sentence appears incomplete or cut off.")

        brief_l = (primary_topic or "").lower()
        thanks_intent = any(
            x in brief_l
            for x in ("good", "great", "thanks", "thank", "reply", "feedback", "liked")
        )
        if thanks_intent and not any(
            x in text.lower()
            for x in ("thank", "glad", "appreciate", "happy", "means a lot", "kind")
        ):
            # soft — only flag if also pitching
            if pitch_hits:
                issues.append(
                    "Comment should acknowledge the positive feedback naturally."
                )

        # Score: comments are short — high score when clean and on-length
        if not issues and 1 <= word_count <= 120:
            dim = {
                "content_quality": 92,
                "seo_compliance": 90,  # N/A for comments — do not punish
                "brand_alignment": 88,
                "structure": 90,
                "factual_grounding": 90,  # N/A — do not require stats
                "natural_voice": 90,
                "cta_effectiveness": 90,  # N/A — do not require brand CTA
            }
            feedback.append("Natural comment reply; length and intent look good.")
            score = self._calculate_score(dim)
            return {
                "score": score,
                "status": "PASS",
                "needs_revision": False,
                "feedback": feedback,
                "issues": [],
                "rewrite_instruction": "",
                "dimension_scores": dim,
                "revision_number": revision_count + 1,
            }

        dim = {
            "content_quality": 55 if issues else 80,
            "seo_compliance": 85,
            "brand_alignment": 60 if pitch_hits else 80,
            "structure": 70,
            "factual_grounding": 85,
            "natural_voice": 65 if any("incomplete" in i.lower() for i in issues) else 80,
            "cta_effectiveness": 85,
        }
        score = self._calculate_score(dim)
        rewrite = (
            "Rewrite as a short natural social comment reply only. "
            "Match the user's intent (e.g. thank them for saying the article is good). "
            "No hashtags, no SEO keywords, no research stats, no hard CTA / sales pitch. "
        )
        if target_n:
            rewrite += f"Use approximately {target_n} words. "
        return {
            "score": score,
            "status": "FAIL",
            "needs_revision": True,
            "feedback": feedback,
            "issues": issues,
            "rewrite_instruction": rewrite.strip(),
            "dimension_scores": dim,
            "revision_number": revision_count + 1,
        }

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
        weights = getattr(self, "_active_weights", None) or DIMENSION_WEIGHTS
        weighted = sum(
            dimension_scores.get(dim, 0) * weight
            for dim, weight in weights.items()
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
            "remove keyword stuffing — use primary terms sparingly where natural; "
            if str(content_type).lower() in ("blog", "article")
            else "do not keyword-stuff; "
        )
        cta_bit = (
            "close with the brand CTA verbatim once only (specific action, not 'reach out today'); "
            if any("BRAND_CTA_MISMATCH" in str(i) for i in issues)
            else "keep a natural close with at most one brand CTA; "
        )
        base = (
            f"Revise to reach an overall score of at least {PASS_THRESHOLD}. "
            f"Priority dimension: {lowest}. "
            "Do NOT add statistics just to look authoritative — keep only on-brief, "
            "section-relevant proof from research; drop ornamental numbers; "
            "remove absolute uncited industry claims; "
            "cut repeated thesis statements across sections (say the core idea once); "
            "for abstract claims that remain, demonstrate in a real business scene "
            "instead of only explaining; "
            f"{secondary_bit}"
            "complete every sentence; "
            f"{cta_bit}"
            "write currency as USD amounts without the $ character; "
            "rewrite in a natural human voice — ban Moreover/Furthermore/In conclusion/"
            "it's worth noting/leverage/cutting-edge/game-changer/Picture a…/Imagine a…/"
            "future-ready/changes the game/the payoff is clear; "
            "vary sentence length; "
            "use full forms only (you are / it is / I would — never you're / it's / I'd); "
            "remove em/en dashes from body copy; "
            "prefer fewer, sharper examples over formulaic claim→example→transition blocks; "
            "keep natural_voice above 80."
        )
        humanize_bits = [
            str(i) for i in issues if str(i).startswith("HUMANIZE_")
        ]
        if humanize_bits:
            base = (
                f"{base} HUMANIZE FAIL — fix before anything else: "
                + "; ".join(humanize_bits[:4])
            )
        evidence_bits = [
            str(i) for i in issues if str(i).startswith("EVIDENCE_")
        ]
        if evidence_bits:
            base = (
                f"{base} EVIDENCE FAIL — remove unsupported figures/"
                "overclaims; use only Evidence Ledger facts: "
                + "; ".join(evidence_bits[:3])
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
