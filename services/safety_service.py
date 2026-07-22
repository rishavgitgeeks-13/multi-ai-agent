"""
Content Safety & Policy Service
===============================

Hard gate used by the Manager (and fallback checks by Review / Writer).

Responsibilities:
- Block harmful / illegal / abusive / hate / religious-conflict prompts
- Allow defensive / awareness topics (e.g. phishing protection)
- Lock the primary topic so later agents do not invert meaning
- Extract user constraints (especially target word count)
- Log policy violations for audit (reputation / abuse monitoring)

This service never generates content.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings

logger = logging.getLogger(__name__)

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_VIOLATION_LOG = _LOG_DIR / "safety_violations.jsonl"

REFUSAL_MESSAGE = (
    "We cannot create content on this topic. "
    "Requests involving sexual-violence how-tos, child exploitation, crime how-tos "
    "(fraud, scams, theft, hacking for harm), hate or discrimination, "
    "or religion content that may inflame social conflict are not allowed. "
    "Educational awareness and statistics topics are allowed — for example "
    "crime/case statistics for societal education, phishing protection, "
    "how trusted nanny/childcare agencies screen differently, NRI fraud awareness, "
    "or balanced AI-market / bubble education. "
    "Please submit a different, appropriate topic."
)

# Educational / defensive awareness — checked BEFORE hard blocks.
# Harm how-tos and graphic exploitation remain blocked when awareness intent is absent.
_DEFENSIVE_ALLOW_PATTERNS: List[re.Pattern[str]] = [
    re.compile(p, re.I)
    for p in [
        # Cyber / phishing / scam protection
        r"\b(protect|protection|prevent|prevention|defend|defense|defence|"
        r"aware|awareness|safe|safety|secure|security|avoid|spot|detect|"
        r"recognis[ea]|recognize|report)\b.{0,60}\b"
        r"(phish|phishing|scam|fraud|hack|hacking|cyber.?attack|malware|"
        r"ransomware|identity.?theft|social.?engineering)\b",
        r"\b(how\s+to\s+)?(protect|stay\s+safe|keep\s+safe|defend)\b.{0,40}\b"
        r"(from|against|online|cyber)\b",
        r"\bphishing\s+(awareness|prevention|protection|safety)\b",
        r"\b(cybersecurity|information\s+security|data\s+protection)\b",
        r"\bhow\s+to\s+(spot|identify|recognis[ea]|avoid)\s+(a\s+)?(scam|phish|fraud)\b",
        # Childcare / nanny agency trust & safety education (not graphic crime content)
        r"\b(how\s+)?(good|trusted|responsible|reputable)\s+"
        r"(nanny|childcare|caregiver)?\s*(agenc(y|ies)|providers?)\b.{0,80}\b"
        r"(screen|vet|train|monitor|differently|better|safety|red\s*flags|hire|hiring)\b",
        r"\b(what\s+)?(responsible|good|trusted)\s+(agencies|providers)\s+"
        r"(do|should)\s+(differently|better)\b",
        r"\b(vet|screen|screening|train|training|monitor|background\s*check|"
        r"red\s*flags|hire|hiring)\b.{0,60}\b"
        r"(nanny|nannies|caregiver|babysitter|childcare|agency|agencies)\b",
        r"\b(nanny|nannies|caregiver|babysitter|childcare)\s+"
        r"(safety|trust|screening|standards|vetting|quality)\b",
        r"\b(parents?|families)\b.{0,50}\b(vet|check|choose|hire|select)\b.{0,40}\b"
        r"(nanny|caregiver|agency|childcare)\b",
        r"\b(after|following)\s+(recent\s+)?(childcare|nanny|caregiver).{0,50}\b"
        r"(safety|incident|concern|news|trend)\b",
        r"\b(childcare|nanny)\s+(safety|trust)\s+(awareness|tips|guide|checklist)\b",
        # Prevention / awareness articles that mention abuse/incidents + how to avoid
        # (stats OK; graphic exploitation still hard-blocked when no prevention framing)
        r"\b(nanny|nannies|childcare|caregiver|babysitter|kinvo)\b[\s\S]{0,400}\b"
        r"(abus\w*|assault|harm|incident|cases?)\b[\s\S]{0,400}\b"
        r"(avoid|prevent|prevention|protect|protection|safety|screen|vet|awareness|"
        r"help\s+avoid|red\s*flags|how\s+.+\s+help)\b",
        r"\b(avoid|prevent|prevention|protect|protection|safety|awareness|screen|vet|"
        r"help\s+avoid|red\s*flags)\b[\s\S]{0,400}\b"
        r"(nanny|nannies|childcare|caregiver|babysitter|kinvo)\b[\s\S]{0,200}\b"
        r"(abus\w*|assault|harm|incident|cases?)\b",
        r"\b(nanny|nannies|childcare|caregiver|babysitter|kinvo)\b[\s\S]{0,500}\b"
        r"(help\s+avoid|prevent|prevention|protect|protection|awareness|safety)\b",
        # NRI fraud awareness (not scam how-to)
        r"\b(nri|nris|non[-\s]?resident\s+indians?)\b.{0,80}\b"
        r"(scam|fraud|aware|awareness|protect|protection|avoid|spot|safe|safety)\b",
        r"\b(protect|spot|avoid|awareness|prevent).{0,50}\b(nri|local)\b.{0,40}\b"
        r"(scam|fraud)\b",
        # AI bubble / investor education
        r"\b(ai|artificial\s+intelligence)\s+bubble\b",
        r"\b(short[-\s]?term|long[-\s]?term)\s+invest(or|ment|ing)?.{0,60}\b"
        r"(ai|bubble|artificial\s+intelligence)\b",
        r"\b(ai|tech)\s+(stock|market|invest).{0,50}\b"
        r"(bubble|risk|discount|volatility|burst)\b",
    ]
]

# Geography is detected from the USER PROMPT only — never forced by brand.
_GEO_HINT_RE = re.compile(
    r"\b("
    r"india|indian|delhi|ncr|gurgaon|gurugram|mumbai|bangalore|bengaluru|"
    r"hyderabad|chennai|kolkata|pune|noida|pocso|ncrb|"
    r"united\s+states|u\.s\.a\.?|u\.s\.|usa|america|american|"
    r"united\s+kingdom|u\.k\.|uk|britain|british|"
    r"canada|canadian|australia|australian|uae|dubai|singapore|"
    r"germany|german|france|french|japan|japanese"
    r")\b",
    re.I,
)

# Never allow-bypass these (CSAM / child-sex how-tos).
_NEVER_ALLOW_RE = re.compile(
    r"\b(csam|pedophil|paedophil|child\s*porn|underage\s*sex|minor\s*sex|"
    r"how\s+to\s+.{0,40}\b(rape|molest|abuse)\s+(a\s+)?(child|kid|infant|baby|minor))\b",
    re.I,
)

_HARD_BLOCK_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    (
        "child_exploitation",
        re.compile(
            r"\b("
            r"csam|underage\s*sex|minor\s*sex|pedophil|paedophil|"
            r"child\s*(sexual\s*)?(porn|pornography|exploitat)|"
            r"how\s+to\s+.{0,40}\b(abuse|assault|molest|rape)\s+(a\s+)?(child|kid|infant|baby|minor)\b"
            r")",
            re.I,
        ),
    ),
    (
        "graphic_child_harm",
        re.compile(
            r"("
            r"\bwashing\s*machine\b.{0,80}\b(child|kid|infant|baby|nanny)\b|"
            r"\b(child|kid|infant|baby)\b.{0,80}\bwashing\s*machine\b|"
            r"\b(tissue|cloth)\b.{0,40}\b(mouth|face)\b.{0,60}\b(child|kid|infant|baby)\b|"
            r"\b(lock|locked|shut)\b.{0,30}\b(bathroom|room)\b.{0,50}\b(child|kid|infant|baby)\b|"
            r"\b(nanny|caregiver)\b.{0,60}\b(lock|locked)\b.{0,40}\b(bathroom|room)\b.{0,40}\b(child|kid|baby)\b"
            r")",
            re.I,
        ),
    ),
    (
        # Instructional / erotic depiction — NOT bare "rape statistics" / awareness
        "sexual_violence",
        re.compile(
            r"\b("
            r"how\s+to\s+(rape|molest|sexually\s+(assault|abuse)|force(?:d)?\s+sex)|"
            r"how\s+to\s+(have|get)\s+sex\b|"
            r"how\s+(an?\s+)?(owner|employer|boss).{0,40}\bsexually\s+(harass|assault|abuse)|"
            r"(write|create|generate|describe).{0,40}\b(rape|sexual\s*assault)\s+(scene|story|erotica)|"
            r"(guide|tutorial|tips)\s+(to|on|for)\s+(rape|molest|having\s+sex)|"
            r"rape\s+(someone|her|him|a\s+(woman|girl|child|kid))\b|"
            r"non[-\s]?consensual\s*sex\s+(guide|howto|how\s+to)|"
            r"force[d]?\s*sex\s+(guide|howto|how\s+to)"
            r")",
            re.I,
        ),
    ),
    (
        "violent_crime_howto",
        re.compile(
            r"\b("
            r"how\s+to\s+(murder|kill\s+someone|torture|assassinate)|"
            r"how\s+to\s+(make|build)\s+a\s+bomb|"
            r"school\s+shooting\s+(guide|howto|how\s+to)"
            r")",
            re.I,
        ),
    ),
    (
        "fraud_scam_howto",
        re.compile(
            r"\b("
            r"how\s+to\s+.{0,40}\b(commit\s+)?(fraud|scam|phish|defraud)\b|"
            r"how\s+to\s+(scam|con|defraud)\s+(people|someone|victims?)|"
            r"how\s+to\s+.{0,40}\b(run\s+a\s+scam|phishing\s+attack|identity\s+theft)\b|"
            r"run\s+a\s+scam\b|"
            r"(create|make)\s+fake\s+(kyc|invoice|cheque|check)\b|"
            r"card\s+skimm\w*|"
            r"ponzi\s+(scheme|scam)\s+(how\s+to|guide|tutorial)"
            r")",
            re.I,
        ),
    ),
    (
        "theft_howto",
        re.compile(
            r"\b("
            r"how\s+to\s+(steal|rob|pickpocket|shoplift|hotwire)\b|"
            r"how\s+to\s+(commit\s+)?(robbery|burglary)\b|"
            r"how\s+to\s+break\s+into\b"
            r")",
            re.I,
        ),
    ),
    (
        "hacking_harm_howto",
        re.compile(
            r"\b("
            r"how\s+to\s+hack\b|"
            r"hack\s+into\s+(an?\s+)?(account|bank|wifi|password|system|network|email)\b|"
            r"how\s+to\s+(crack|break\s+into)\s+(a\s+)?(password|wifi|account|system)\b|"
            r"how\s+to\s+(create|build|write|deploy|spread|launch)\s+(a\s+)?"
            r"(ransomware|malware|virus|trojan|keylogger)\b|"
            r"how\s+to\s+(run|perform|launch|carry\s+out)\s+(a\s+)?"
            r"(ddos(\s+attack)?|credential\s+stuffing)\b|"
            r"how\s+to\s+steal\s+(passwords?|credentials|data)\b|"
            r"steal\s+(passwords?|credentials)\s+(from|using|via|with)\b"
            r")",
            re.I,
        ),
    ),
    (
        "hate_discrimination",
        re.compile(
            r"\b(hate\s+(speech|crime)|genocide|ethnic\s+cleansing|"
            r"(inferior|subhuman).{0,30}(caste|race|gender|religion|region)|"
            r"(all|those)\s+(women|men|muslims|hindus|christians|dalits|"
            r"blacks|whites)\s+(are|should)\s+(stupid|evil|vermin|die))\b",
            re.I,
        ),
    ),
    (
        "self_harm",
        re.compile(
            r"\b(how\s+to\s+)?(commit\s+suicide|kill\s+myself|"
            r"self[-\s]?harm\s+method)\b",
            re.I,
        ),
    ),
    (
        "religious_conflict",
        re.compile(
            r"\b((attack|insult|mock|hate|destroy|wipe\s+out)\s+"
            r"(hindu|muslim|christian|sikh|jew|jewish|buddhist|islam|"
            r"hinduism|christianity|judaism|religion)|"
            r"(hindu|muslim|christian|sikh|jew).{0,40}"
            r"(are\s+(terrorists|evil|animals|inferior)|should\s+(die|be\s+banned))|"
            r"why\s+(hinduism|islam|christianity|sikhism)\s+is\s+(fake|evil|wrong)|"
            r"religious\s+(war|hatred|violence)\s+(guide|how\s+to)|"
            r"convert\s+(or\s+)?(die|force)|forced\s+conversion)\b",
            re.I,
        ),
    ),
]

_SENSITIVE_TERMS = re.compile(
    r"\b(rape|sexual\s*assault|child\s*abuse|molest|csam|pedophil|"
    r"scam|fraud|hack|ransomware|steal|robbery|genocide|"
    r"caste\s*hatred|religious\s*hatred)\b",
    re.I,
)

_WORD_COUNT_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\b(?:exactly\s+)?(\d{1,5})\s*[\-]?\s*words?\b", re.I),
    re.compile(r"\b(?:word\s*count|length|limit)\s*(?:of|is|=|:)?\s*(\d{1,5})\b", re.I),
    re.compile(
        r"\b(?:under|below|max(?:imum)?|at\s+most|no\s+more\s+than)\s+(\d{1,5})\s*words?\b",
        re.I,
    ),
    re.compile(r"\b(?:about|around|approx(?:imately)?|~)\s*(\d{1,5})\s*words?\b", re.I),
    re.compile(r"\bi\s+want\s+(?:a\s+)?(\d{1,5})\s*words?\b", re.I),
]


class SafetyService:
    """Policy evaluation, constraint extraction, and violation logging."""

    def __init__(self) -> None:
        self._openai = None
        if settings.OPENAI_API_KEY:
            try:
                from openai import OpenAI

                self._openai = OpenAI(api_key=settings.OPENAI_API_KEY)
            except Exception as exc:
                logger.warning("SafetyService: OpenAI unavailable: %s", exc)

    def evaluate_request(
        self,
        user_input: str,
        *,
        request_id: str = "",
        session_id: str = "",
        brand: Optional[str] = None,
        content_type: str = "",
        source: str = "manager",
    ) -> Dict[str, Any]:
        text = (user_input or "").strip()
        constraints = self.extract_constraints(text)
        primary_topic = self._derive_primary_topic(text, constraints)

        if not text:
            return self._result(
                allowed=False,
                category="empty",
                reason="Empty user input",
                primary_topic="",
                constraints=constraints,
            )

        # Absolute never-allow (CSAM / child-sex how-tos)
        if _NEVER_ALLOW_RE.search(text):
            decision = self._result(
                allowed=False,
                category="child_exploitation",
                reason="Matched never-allow child exploitation / CSAM policy",
                primary_topic=primary_topic,
                constraints=constraints,
            )
            self.log_violation(
                user_input=text,
                decision=decision,
                request_id=request_id,
                session_id=session_id,
                brand=brand,
                content_type=content_type,
                source=source,
                stage="request",
            )
            return decision

        # Stats / awareness / prevention — allow and keep the user's real topic
        if self._is_educational_or_awareness_intent(text) or self._is_defensive_allow(
            text
        ):
            framed_topic = self._frame_safe_primary_topic(
                text, primary_topic, brand=brand
            )
            return self._result(
                allowed=True,
                category="defensive_awareness",
                reason=(
                    "Educational / awareness / stats topic allowed "
                    "(primary user topic preserved; geography follows user prompt)"
                ),
                primary_topic=framed_topic,
                constraints=constraints,
                defensive_allow=True,
            )

        category, reason = self._match_hard_block(text)
        if category:
            decision = self._result(
                allowed=False,
                category=category,
                reason=reason,
                primary_topic=primary_topic,
                constraints=constraints,
            )
            self.log_violation(
                user_input=text,
                decision=decision,
                request_id=request_id,
                session_id=session_id,
                brand=brand,
                content_type=content_type,
                source=source,
                stage="request",
            )
            return decision

        llm_decision = self._classify_with_llm(text)
        if llm_decision is not None:
            allowed = bool(llm_decision.get("allowed", True))
            # Prefer educational allow when classifier is unsure but intent is stats/awareness
            if (
                not allowed
                and self._is_educational_or_awareness_intent(text)
                and not _NEVER_ALLOW_RE.search(text)
            ):
                allowed = True
                category = "defensive_awareness"
                reason = "Reclassified as educational/awareness/stats intent"
            else:
                category = str(
                    llm_decision.get("category")
                    or ("safe" if allowed else "policy_violation")
                )
                reason = str(llm_decision.get("reason") or "")
            if llm_decision.get("primary_topic") and allowed:
                # Prefer user-derived topic; only use LLM topic if it did not diverge
                llm_topic = str(llm_decision["primary_topic"]).strip()
                if llm_topic and not self._topics_diverged(primary_topic, llm_topic):
                    primary_topic = llm_topic
            if allowed:
                primary_topic = self._frame_safe_primary_topic(
                    text, primary_topic, brand=brand
                )
            decision = self._result(
                allowed=allowed,
                category=category,
                reason=reason
                or (
                    "Allowed by policy classifier"
                    if allowed
                    else "Blocked by policy classifier"
                ),
                primary_topic=primary_topic,
                constraints=constraints,
                defensive_allow=allowed
                and self._is_educational_or_awareness_intent(text),
            )
            if not allowed:
                self.log_violation(
                    user_input=text,
                    decision=decision,
                    request_id=request_id,
                    session_id=session_id,
                    brand=brand,
                    content_type=content_type,
                    source=source,
                    stage="request_llm",
                )
            return decision

        return self._result(
            allowed=True,
            category="safe",
            reason="No policy violation detected",
            primary_topic=self._frame_safe_primary_topic(
                text, primary_topic, brand=brand
            ),
            constraints=constraints,
        )

    def evaluate_draft(
        self,
        draft: str,
        *,
        primary_topic: str = "",
        user_input: str = "",
        request_id: str = "",
        session_id: str = "",
        brand: Optional[str] = None,
        content_type: str = "",
        source: str = "review",
    ) -> Dict[str, Any]:
        text = (draft or "").strip()
        if not text:
            return self._result(
                allowed=True,
                category="empty_draft",
                reason="Empty draft",
                primary_topic=primary_topic,
                constraints={},
            )

        category, reason = self._match_hard_block(text)
        if category:
            decision = self._result(
                allowed=False,
                category=category,
                reason=f"Draft contains blocked content: {reason}",
                primary_topic=primary_topic,
                constraints={},
            )
            self.log_violation(
                user_input=user_input or primary_topic,
                decision=decision,
                request_id=request_id,
                session_id=session_id,
                brand=brand,
                content_type=content_type,
                source=source,
                stage="draft",
                extra={"draft_preview": text[:400]},
            )
            return decision

        if self._looks_like_topic_inversion(user_input or primary_topic, text):
            decision = self._result(
                allowed=False,
                category="topic_inversion",
                reason="Draft inverted or diverted from the primary user topic",
                primary_topic=primary_topic,
                constraints={},
            )
            self.log_violation(
                user_input=user_input or primary_topic,
                decision=decision,
                request_id=request_id,
                session_id=session_id,
                brand=brand,
                content_type=content_type,
                source=source,
                stage="draft_fidelity",
                extra={"draft_preview": text[:400]},
            )
            return decision

        return self._result(
            allowed=True,
            category="safe",
            reason="Draft passed safety fallback",
            primary_topic=primary_topic,
            constraints={},
        )

    def extract_constraints(self, user_input: str) -> Dict[str, Any]:
        text = user_input or ""
        constraints: Dict[str, Any] = {
            "target_word_count": None,
            "word_count_flexible": True,
            "raw_length_mentions": [],
        }
        for pat in _WORD_COUNT_PATTERNS:
            for match in pat.finditer(text):
                try:
                    n = int(match.group(1))
                except (IndexError, ValueError):
                    continue
                if 1 <= n <= 50000:
                    constraints["raw_length_mentions"].append(n)
                    if constraints["target_word_count"] is None:
                        constraints["target_word_count"] = n
                        if re.search(rf"\bexactly\s+{n}\s*words?\b", text, re.I):
                            constraints["word_count_flexible"] = False
        return constraints

    def log_violation(
        self,
        *,
        user_input: str,
        decision: Dict[str, Any],
        request_id: str = "",
        session_id: str = "",
        brand: Optional[str] = None,
        content_type: str = "",
        source: str = "",
        stage: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "session_id": session_id,
            "brand": brand,
            "content_type": content_type,
            "source": source,
            "stage": stage,
            "category": decision.get("category"),
            "reason": decision.get("reason"),
            "user_input_preview": (user_input or "")[:500],
            "user_input_length": len(user_input or ""),
            "sensitive_terms_detected": bool(_SENSITIVE_TERMS.search(user_input or "")),
        }
        if extra:
            record.update(extra)

        logger.warning(
            "SAFETY_VIOLATION | category=%s | session=%s | request=%s | brand=%s | reason=%s | preview=%s",
            record["category"],
            session_id,
            request_id,
            brand,
            record["reason"],
            record["user_input_preview"][:120],
        )

        try:
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
            with _VIOLATION_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.error("Failed to write safety violation log: %s", exc)

    @staticmethod
    def _result(
        *,
        allowed: bool,
        category: str,
        reason: str,
        primary_topic: str,
        constraints: Dict[str, Any],
        defensive_allow: bool = False,
    ) -> Dict[str, Any]:
        return {
            "allowed": allowed,
            "blocked": not allowed,
            "category": category,
            "reason": reason,
            "message": "" if allowed else REFUSAL_MESSAGE,
            "primary_topic": primary_topic,
            "user_constraints": constraints or {},
            "defensive_allow": defensive_allow,
        }

    @staticmethod
    def _is_defensive_allow(text: str) -> bool:
        return any(p.search(text) for p in _DEFENSIVE_ALLOW_PATTERNS)

    @classmethod
    def _is_educational_or_awareness_intent(cls, text: str) -> bool:
        """
        Allow stats / societal education / prevention framing.
        Does NOT allow clear harmful how-tos (those stay hard-blocked).
        """
        t = text or ""
        if not t.strip() or _NEVER_ALLOW_RE.search(t):
            return False
        # Clear instructional harm stays blocked even if "stats" appears nearby.
        howto_hit, _ = cls._match_hard_block(t)
        if howto_hit in {
            "sexual_violence",
            "violent_crime_howto",
            "fraud_scam_howto",
            "theft_howto",
            "hacking_harm_howto",
            "child_exploitation",
            "graphic_child_harm",
            "self_harm",
        }:
            return False

        has_edu = bool(
            re.search(
                r"\b("
                r"stats?|statistics|statistical|data|figures?|numbers?|"
                r"cases?|incidents?|reports?|reporting|prevalence|rate|"
                r"awareness|societal|education(al)?|public\s+health|"
                r"trends?|year[-\s]?wise|state[-\s]?wise|overview|"
                r"what\s+are\s+the\s+(stats|statistics|figures|numbers)"
                r")\b",
                t,
                re.I,
            )
        )
        has_prevention = cls._has_prevention_intent(t)
        has_sensitive = bool(_SENSITIVE_TERMS.search(t)) or cls._is_childcare_sensitive(
            t
        )
        # Stats/awareness about sensitive societal topics, or prevention + sensitive.
        if has_edu and (has_sensitive or has_prevention):
            return True
        if has_prevention and has_sensitive:
            return True
        return False

    @staticmethod
    def _has_prevention_intent(text: str) -> bool:
        """Broader prevention signal used for awareness / educational allows."""
        t = text or ""
        return bool(
            re.search(
                r"\b(prevent|prevention|avoid|protect|protection|awareness|safety|"
                r"screen|screening|vet|vetting|red\s*flags|checklist|"
                r"help\s+avoid|how\s+.+\s+help|trusted\s+agenc)\b",
                t,
                re.I,
            )
        )

    @staticmethod
    def _strip_brand_prefix(text: str) -> str:
        return re.sub(
            r"^\[\s*brand\s*:[^\]]+\]\s*",
            "",
            (text or "").strip(),
            flags=re.I,
        ).strip()

    @staticmethod
    def _user_geography_tokens(text: str) -> List[str]:
        """Geography tokens from the user prompt only (never inferred from brand)."""
        found = [m.group(0) for m in _GEO_HINT_RE.finditer(text or "")]
        # Preserve order, de-dupe case-insensitively
        seen = set()
        out: List[str] = []
        for tok in found:
            key = tok.lower()
            if key not in seen:
                seen.add(key)
                out.append(tok)
        return out[:6]

    @staticmethod
    def _is_childcare_sensitive(text: str) -> bool:
        t = (text or "").lower()
        return bool(
            re.search(
                r"\b(nanny|nannies|childcare|caregiver|babysitter)\b",
                t,
            )
            and re.search(
                r"\b(abus\w*|assault|molest|rape|harm|incident|exploit)\b",
                t,
            )
        )

    @classmethod
    def _frame_safe_primary_topic(
        cls,
        text: str,
        fallback: str,
        brand: Optional[str] = None,
    ) -> str:
        """
        Keep the user's real topic (do NOT replace with a generic screening essay).
        Append soft framing so Writer/Research stay on-brief without graphic junk.
        Geography follows the user prompt only — brand never forces a market.
        """
        del brand  # Brand tone/CTA live in brands.yaml; do not force geography.
        base = cls._strip_brand_prefix(fallback or text or "")
        if not base:
            base = cls._strip_brand_prefix(text or "")[:300]
        # Prefer the cleaned user ask as the core topic (not a rewritten subject).
        user_core = cls._strip_brand_prefix(text or "")
        if user_core and len(user_core) >= 20:
            # If LLM already rewrote away from the user ask, restore user core.
            if cls._topics_diverged(user_core, base):
                base = user_core[:400]

        extras: List[str] = []
        geo_tokens = cls._user_geography_tokens(text) or cls._user_geography_tokens(
            base
        )
        if geo_tokens:
            geo_list = ", ".join(geo_tokens)
            extras.append(
                f"Geography: follow the user prompt ({geo_list}) — use matching "
                f"market data and sources; do not substitute unrelated countries."
            )

        if cls._is_childcare_sensitive(text) or cls._is_childcare_sensitive(base):
            extras.append(
                "Safety framing: prevention/awareness for parents — high-level "
                "statistics and red flags only; no graphic incident detail, "
                "no sensational crime storytelling, no sexual content."
            )

        if extras:
            return f"{base[:420].rstrip()} | {' '.join(extras)}"[:700]
        return base[:500]

    @staticmethod
    def _topics_diverged(user_topic: str, candidate: str) -> bool:
        """True when candidate dropped core tokens from the user ask (junk rewrite)."""
        def tokens(s: str) -> set:
            stop = {
                "write", "article", "about", "the", "and", "for", "with", "from",
                "that", "this", "should", "have", "been", "where", "between",
                "add", "an", "angle", "on", "how", "can", "help", "such",
            }
            return {
                w
                for w in re.findall(r"[a-z0-9]{3,}", (s or "").lower())
                if w not in stop
            }

        u = tokens(user_topic)
        c = tokens(candidate)
        if not u:
            return False
        overlap = len(u & c) / max(1, len(u))
        # If the candidate kept <35% of user tokens, it likely replaced the topic.
        return overlap < 0.35

    @staticmethod
    def _lock_awareness_topic(text: str, fallback: str) -> str:
        """Backward-compatible alias — preserves topic via framing."""
        return SafetyService._frame_safe_primary_topic(text, fallback)

    @staticmethod
    def _match_hard_block(text: str) -> Tuple[str, str]:
        for category, pattern in _HARD_BLOCK_PATTERNS:
            if pattern.search(text):
                return category, f"Matched hard policy category: {category}"
        return "", ""

    @staticmethod
    def _derive_primary_topic(text: str, constraints: Dict[str, Any]) -> str:
        cleaned = text
        for pat in _WORD_COUNT_PATTERNS:
            cleaned = pat.sub(" ", cleaned)
        cleaned = re.sub(
            r"\b(i want|please|write|create|generate|make)\b",
            " ",
            cleaned,
            flags=re.I,
        )
        cleaned = re.sub(r"^\[\s*brand\s*:[^\]]+\]\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,;:-")
        return cleaned[:300] if cleaned else text[:300]

    @staticmethod
    def _looks_like_topic_inversion(user_topic: str, draft: str) -> bool:
        topic = (user_topic or "").lower()
        body = (draft or "").lower()
        if not topic or not body:
            return False
        victim_caregiver = bool(
            re.search(r"\b(nanny|caregiver|babysitter|maid|domestic\s+worker)\b", topic)
            and re.search(r"\b(assault|abuse|harass|rape|molest|victim)\b", topic)
        )
        draft_caregiver_abuses_child = bool(
            re.search(
                r"\b(nanny|caregiver|babysitter).{0,80}\b(abuse|assault|molest|rape)"
                r".{0,40}\b(child|kid|infant|baby|minor)\b",
                body,
            )
            or re.search(
                r"\b(child|kid|infant|baby|minor).{0,80}\b(abuse|assault|molest)"
                r".{0,40}\b(nanny|caregiver|babysitter)\b",
                body,
            )
        )
        return victim_caregiver and draft_caregiver_abuses_child

    def _classify_with_llm(self, text: str) -> Optional[Dict[str, Any]]:
        if not self._openai:
            return None
        if not _SENSITIVE_TERMS.search(text) and len(text) < 40:
            return None

        system = (
            "You are a strict content-policy classifier for a brand content platform. "
            "Return ONLY valid JSON with keys: "
            "allowed (bool), category (string), reason (string), primary_topic (string).\n"
            "Judge INTENT, not bare keywords.\n"
            "BLOCK (allowed=false) clear harmful how-tos / non-educational asks, including: "
            "'how to rape', 'how to have sex', 'how to scam people', crime/fraud/theft/hacking "
            "instructionals, CSAM/child-sex content, graphic child-harm retellings for "
            "entertainment, hate/discrimination, religion content that inflames conflict, "
            "self-harm instructions.\n"
            "ALLOW (allowed=true) educational / awareness / prevention intent, including: "
            "statistics and societal education (e.g. 'stats of rape cases in India'); "
            "awareness and prevention articles that cite high-level case data; "
            "phishing/scam/cyber protection; parent childcare safety / nanny screening / "
            "red flags; brand prevention angles (e.g. how a verified agency helps avoid "
            "incidents); NRI fraud awareness; balanced AI bubble education. "
            "Do not invent geography — keep whatever market the user named "
            "(India, US, UK, etc.). Never assume a market from the brand alone.\n"
            "CRITICAL for primary_topic when allowed: preserve the user's actual request "
            "(geography, years, data asks, brand angle). Do NOT replace it with a generic "
            "screening-only essay. You may append a short safety note, but keep the core ask."
        )
        user = f"Classify this content request:\n\n{text}"

        try:
            resp = self._openai.chat.completions.create(
                model=settings.OPENAI_MODEL,
                max_tokens=400,
                temperature=0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            if not isinstance(data, dict) or "allowed" not in data:
                return None
            return data
        except Exception as exc:
            logger.warning("Safety LLM classify failed (non-fatal): %s", exc)
            return None


safety_service = SafetyService()
