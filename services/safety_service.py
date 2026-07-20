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
    "Requests involving sexual violence, child abuse, crime how-tos "
    "(fraud, scams, theft, hacking for harm), hate or discrimination, "
    "or religion content that may inflame social conflict are not allowed. "
    "Educational awareness topics are allowed — for example phishing protection, "
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

_HARD_BLOCK_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    (
        "child_exploitation",
        re.compile(
            r"\b(child\s*(sexual\s*)?(abuse|assault|porn|pornography|exploitat)|"
            r"csam|underage\s*sex|minor\s*sex|pedophil|paedophil|"
            r"nanny\s+.{0,40}\b(abuse|assault|molest|rape).{0,40}\b(child|kid|infant|baby)|"
            r"(abuse|assault|molest|rape).{0,40}\b(child|kid|infant|baby|minor))\b",
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
        "sexual_violence",
        re.compile(
            r"\b("
            r"rape|raping|rapist|"
            r"sexual\s*assault\w*|sexually\s*assault\w*|"
            r"sexual\s*abuse\w*|sexually\s*abuse\w*|"
            r"molest\w*|molestation|"
            r"non[-\s]?consensual\s*sex|force[d]?\s*sex"
            r")\b",
            re.I,
        ),
    ),
    (
        "violent_crime_howto",
        re.compile(
            r"\b(how\s+to\s+)?(murder|kill\s+someone|torture|assassinate|"
            r"make\s+a\s+bomb|build\s+a\s+bomb|school\s+shooting)\b",
            re.I,
        ),
    ),
    (
        "fraud_scam_howto",
        re.compile(
            # Block scam/fraud *how-tos* only — not educational mentions of fraud/phishing.
            r"\b("
            r"how\s+to\s+.{0,40}\b(commit\s+)?(fraud|scam|phish|defraud)\b|"
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
            # Block offensive hacking *how-tos* only.
            # Bare words like ransomware / phishing / ddos are normal in cybersecurity
            # startup and awareness articles and must NOT block drafts.
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

        if self._is_defensive_allow(text):
            locked_topic = self._lock_awareness_topic(text, primary_topic)
            return self._result(
                allowed=True,
                category="defensive_awareness",
                reason=(
                    "Educational / awareness topic allowed "
                    "(safe framing locked for downstream agents)"
                ),
                primary_topic=locked_topic,
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
            category = str(
                llm_decision.get("category")
                or ("safe" if allowed else "policy_violation")
            )
            reason = str(llm_decision.get("reason") or "")
            if llm_decision.get("primary_topic"):
                primary_topic = str(llm_decision["primary_topic"]).strip() or primary_topic
            if allowed:
                primary_topic = self._lock_awareness_topic(text, primary_topic)
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
                defensive_allow=allowed and self._is_defensive_allow(text),
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
            primary_topic=primary_topic,
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

    @staticmethod
    def _lock_awareness_topic(text: str, fallback: str) -> str:
        """
        Rewrite awareness prompts into a safe primary topic so Writer/Research
        do not recreate graphic incidents — only education / trust / protection.
        """
        t = (text or "").lower()
        if re.search(
            r"\b(nanny|nannies|childcare|caregiver|babysitter|agenc(?:y|ies))\b",
            t,
        ):
            return (
                "How responsible childcare and nanny agencies screen, train, and "
                "monitor caregivers — and what parents should check so they can "
                "hire with more trust"
            )
        if re.search(r"\b(nri|nris|non[-\s]?resident)\b", t):
            return (
                "How NRIs can spot and avoid local fraud and scams — practical "
                "awareness and protection tips"
            )
        if re.search(r"\b(ai|artificial\s+intelligence)\s+bubble\b", t) or (
            re.search(r"\b(ai|artificial\s+intelligence)\b", t)
            and re.search(r"\b(bubble|burst|short[-\s]?term|long[-\s]?term)\b", t)
        ):
            return (
                "Balanced advice on the AI market: short-term bubble risk versus "
                "long-term AI technology investment opportunities"
            )
        if re.search(
            r"\b(phish|phishing|scam|fraud|hack|cyber|ransomware|malware)\b",
            t,
        ):
            return (
                "How to protect yourself from online scams, phishing, and cyber "
                "attacks — practical awareness tips"
            )
        return (fallback or text)[:300]

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
            "BLOCK (allowed=false) when the user wants content that: "
            "depicts or instructs sexual violence, rape, child abuse/exploitation; "
            "graphic retelling of crimes against children for engagement; "
            "how-to for fraud, scams, theft, hacking for harm; "
            "hate/discrimination by caste, gender, race, region, religion; "
            "religion content that insults faiths or inflames communal conflict; "
            "self-harm instructions; graphic glorification of crime.\n"
            "ALLOW (allowed=true) defensive/educational awareness content such as: "
            "phishing/scam/cyber protection; "
            "how trusted nanny/childcare agencies screen, train, and monitor differently "
            "and parent hiring checklists (without graphic abuse detail); "
            "NRI fraud awareness and protection; "
            "balanced AI bubble / short-term vs long-term investor education.\n"
            "When allowing awareness, primary_topic must be a SAFE educational restatement "
            "(trust, screening, protection, balanced advice) — never a graphic incident narrative."
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
