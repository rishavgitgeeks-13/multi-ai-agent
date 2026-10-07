"""
Final Editor Service
====================

Surgical editorial pass after Review (PASS or force-PASS).

Edits in place — does NOT regenerate the article to re-satisfy SEO quotas.
Fixes: keyword stuffing, thesis repetition, soft B2B clichés, ornamental stats,
duplicate CTAs, formulaic openings.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from config.settings import settings
from services.final_qc import qc_summary, run_final_qc

logger = logging.getLogger(__name__)


class FinalEditorService:
    """Deterministic QC + optional surgical LLM edit."""

    def __init__(self) -> None:
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = settings.model_for_helpers()
        self._temperature = 0.25

    def run(
        self,
        draft: str,
        strategy: Optional[Dict[str, Any]] = None,
        brand_context: Optional[Dict[str, Any]] = None,
        primary_topic: str = "",
        review: Optional[Dict[str, Any]] = None,
        only_flags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        strategy = strategy or {}
        brand_context = brand_context or {}
        review = review or {}
        content_type = str(strategy.get("content_type") or "article").lower()
        cta = str(
            strategy.get("cta")
            or brand_context.get("cta")
            or ""
        ).strip()
        primary = list(strategy.get("keywords") or strategy.get("primary_keywords") or [])
        secondary = list(
            strategy.get("secondary_keywords")
            or (strategy.get("seo") or {}).get("secondary_keywords")
            or []
        )
        # Prefer strategy SEO package if nested
        seo = strategy.get("seo") or {}
        if seo.get("primary_keywords"):
            primary = list(seo.get("primary_keywords") or primary)
        if seo.get("secondary_keywords"):
            secondary = list(seo.get("secondary_keywords") or secondary)
        brand_kws = list(
            seo.get("brand_keywords")
            or brand_context.get("brand_keywords")
            or ((brand_context.get("seo_kit_selected") or {}).get("brand_keywords") or [])
        )
        mode_policy = brand_context.get("mode_policy") or strategy.get("mode_policy") or {}

        evidence_ledger = list(
            strategy.get("evidence_ledger")
            or (brand_context.get("evidence_ledger") if isinstance(brand_context, dict) else None)
            or []
        )
        searcher_questions = list(strategy.get("searcher_questions") or [])

        flags = run_final_qc(
            draft or "",
            primary_keywords=primary,
            secondary_keywords=secondary,
            cta=cta,
            primary_topic=primary_topic or str(strategy.get("title") or ""),
            content_type=content_type,
            mode_policy=mode_policy if isinstance(mode_policy, dict) else {},
            brand_keywords=brand_kws,
            evidence_ledger=evidence_ledger,
            searcher_questions=searcher_questions,
        )
        if only_flags:
            # Surgical single-chip: keep only flags matching requested prefixes/substrings
            wanted = [str(f).strip() for f in only_flags if str(f).strip()]
            filtered = []
            for fl in flags:
                fl_s = str(fl)
                if any(
                    w in fl_s or fl_s.startswith(w) or w.upper() in fl_s.upper()
                    for w in wanted
                ):
                    filtered.append(fl_s)
            # If chip names are category keys, map them
            if not filtered:
                key_map = {
                    "seo_stuff": "QC_SEO_STUFF",
                    "duplicate_cta": "QC_CTA",
                    "thesis_repeat": "QC_THESIS",
                    "b2b_cliche": "QC_B2B",
                    "ornamental_stat": "QC_ORNAMENTAL",
                    "formula": "QC_FORMULA",
                    "mode_h1": "QC_MODE_H1",
                    "demonstrate": "DEMONSTRATE",
                }
                for w in wanted:
                    prefix = key_map.get(w.lower().replace(" ", "_"), w)
                    filtered.extend([fl for fl in flags if prefix in str(fl)])
            flags = filtered or wanted

        summary = qc_summary(flags)
        original = draft or ""
        edited = original
        edited_by_llm = False

        if flags and content_type in ("blog", "article", "linkedin", "email"):
            try:
                edited = self._surgical_edit(
                    draft=original,
                    flags=flags,
                    cta=cta,
                    primary_topic=primary_topic,
                    content_type=content_type,
                )
                if edited and edited.strip() and edited.strip() != original.strip():
                    edited_by_llm = True
                    # Re-scan after edit (informational)
                    flags_after = run_final_qc(
                        edited,
                        primary_keywords=primary,
                        secondary_keywords=secondary,
                        cta=cta,
                        primary_topic=primary_topic,
                        content_type=content_type,
                        mode_policy=mode_policy if isinstance(mode_policy, dict) else {},
                        brand_keywords=brand_kws,
                        evidence_ledger=evidence_ledger,
                        searcher_questions=searcher_questions,
                    )
                    summary = qc_summary(flags_after)
                    summary["flags_before"] = flags
                else:
                    edited = original
            except Exception as exc:
                logger.warning("Final editor LLM pass failed: %s", exc)
                edited = original

        # Deterministic CTA dedupe even if LLM skipped
        if cta and edited:
            edited = self._dedupe_cta(edited, cta)

        # Always scrub AI / B2B clichés (even when Review only warned)
        try:
            from services.text_cleanup import scrub_ai_cliches

            scrubbed = scrub_ai_cliches(edited or "")
            if scrubbed != (edited or ""):
                logger.info("FinalEditor cliche scrub applied")
                edited = scrubbed
        except Exception:
            pass

        # Zero-junk packaging scrub (Sources fragments, vertical hashtags, tool leaks)
        try:
            from services.junk_gate import scrub_packaged_markdown

            edited, scrub_notes = scrub_packaged_markdown(edited or "")
            if scrub_notes:
                logger.info("FinalEditor junk scrub | notes=%s", scrub_notes[:6])
        except Exception:
            pass

        score = int(review.get("score") or 0)
        below_target = bool(review.get("below_target")) or (
            str(review.get("status") or "").upper() == "PASS" and score < 95
        )

        return {
            "draft": edited,
            "final_qc": {
                **summary,
                "edited": edited_by_llm or (edited != original),
                "below_target": below_target,
                "review_score": score,
            },
        }

    def _surgical_edit(
        self,
        draft: str,
        flags: List[str],
        cta: str,
        primary_topic: str,
        content_type: str,
    ) -> str:
        flag_block = "\n".join(f"- {f}" for f in flags[:10])
        cta_line = (
            f"- Keep the brand CTA at most once (prefer Conclusion): {cta}\n"
            if cta
            else "- Keep at most one CTA close.\n"
        )
        prompt = f"""You are a surgical final editor. Edit the {content_type} Markdown below.

GOAL: Fix editorial defects by SUBTRACTION and light rewriting. Do NOT regenerate the article.
Do NOT add SEO keywords, do NOT add new statistics for authority, do NOT expand length.

PRIMARY TOPIC (preserve): {primary_topic or "(unchanged)"}

QC FLAGS TO FIX:
{flag_block}

EDIT RULES:
- Remove keyword stuffing (keep 2–4 natural primary mentions max).
- Delete or merge repeated thesis sentences across sections.
- Replace soft B2B clichés with plain specific language.
  Ban these phrases entirely: "future ready", "future-ready", "not only … but also",
  "changes the game", "the payoff is clear", "unlock growth", "best-in-class",
  "end-to-end solution", "workforce of the future", "digital transformation journey".
- Drop ornamental / off-angle statistics; keep only on-brief proof already present.
- If flagged for brand sales pitch / off-brief: put the USER brief subject first;
  move brand CTA to a single soft close (at most once). Do not turn the piece into a brochure.
- If flagged DEMONSTRATE: rewrite 2–3 abstract ROI/workflow claims into concrete scenes
  (who, tool, what breaks or improves, measurable result). Do not only define the idea.
{cta_line}- Vary cloned section openings only where flagged — do not rewrite every paragraph.
- Preserve headings, factual claims that remain, and Markdown structure.
- No contractions; no dash characters; return ONLY the full edited Markdown.

DRAFT:
{draft[:14000]}
"""
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=8192,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise editorial surgeon. Subtract fluff and stuffing. "
                        "Never invent facts or regenerate from scratch. Return Markdown only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        # Strip accidental fences
        if text.startswith("```"):
            text = re.sub(r"^```(?:markdown|md)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @staticmethod
    def _dedupe_cta(draft: str, cta: str) -> str:
        """Keep the last occurrence of the CTA; neutralize earlier duplicates lightly."""
        cta = (cta or "").strip()
        if not cta or len(cta) < 8:
            return draft
        pattern = re.compile(re.escape(cta), re.I)
        matches = list(pattern.finditer(draft))
        if len(matches) < 2:
            return draft
        # Remove all but the last
        keep_start = matches[-1].start()
        out = []
        last = 0
        for m in matches[:-1]:
            out.append(draft[last : m.start()])
            # Soft replacement that does not hard-sell again
            out.append("talk with the team when you are ready")
            last = m.end()
        out.append(draft[last:])
        # Ensure last CTA spelling matches canonical
        result = "".join(out)
        # Replace last fuzzy match region with canonical CTA if needed
        matches2 = list(pattern.finditer(result))
        if matches2:
            m = matches2[-1]
            result = result[: m.start()] + cta + result[m.end() :]
        return result
