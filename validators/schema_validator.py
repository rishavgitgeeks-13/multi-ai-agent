"""
Schema Validator
================

Validates the data flowing between LangGraph agents against the Pydantic
schemas defined in the schemas/ package.

Called at agent boundaries to catch contract violations early — before
a malformed dict reaches a downstream service and produces a cryptic error.

Three validation layers:

  1. ContentState field validator
     — checks required fields are present and non-empty at each agent handoff

  2. Pydantic model validators
     — validates research_data, strategy, and review dicts against their schemas

  3. Agent boundary validators
     — one function per agent transition, checking the state is ready
       for the next agent to consume

Usage
-----
    from validators.schema_validator import SchemaValidator

    # Validate before the Strategy Agent runs
    result = SchemaValidator.validate_for_strategy(state)
    if not result.ok:
        logger.error("State invalid for strategy: %s", result.errors)
        state["errors"].extend(result.errors)

    # Validate a research_data dict
    result = SchemaValidator.validate_research_data(state["research_data"])

    # Validate a review dict
    result = SchemaValidator.validate_review(state["review"])
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from schemas.research_schema import ResearchData
from schemas.review_schema import ReviewResult
from schemas.strategy_schema import Strategy

logger = logging.getLogger(__name__)

# PASS threshold — must match ReviewService and review_schema
_PASS_THRESHOLD = 70

# Valid enum values for ContentState fields
_VALID_CONTENT_TYPES = {"article", "blog", "linkedin", "email", "carousel"}
_VALID_PLATFORMS = {"website", "linkedin", "email", "x"}
_VALID_OBJECTIVES = {"seo", "engagement", "authority", "leads"}
_VALID_WORKFLOW_STATUSES = {"INIT", "RUNNING", "COMPLETED", "FAILED"}
_VALID_AGENTS = {"manager", "research", "strategy", "writer", "review", "end"}


# ==========================================================================
# Result wrapper
# ==========================================================================


@dataclass
class SchemaValidationResult:
    """
    Result of a schema validation check.
    `ok` is True only when zero errors are found.
    Warnings are non-fatal; errors block the workflow.
    """

    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    data: Any = None

    @classmethod
    def success(cls, data: Any = None, warnings: List[str] = None) -> "SchemaValidationResult":
        return cls(ok=True, data=data, warnings=warnings or [])

    @classmethod
    def failure(cls, errors: List[str], data: Any = None) -> "SchemaValidationResult":
        return cls(ok=False, errors=errors, data=data)

    def log(self, context: str = "") -> None:
        """Log all errors and warnings at appropriate levels."""
        prefix = f"[{context}] " if context else ""
        for err in self.errors:
            logger.error("%sValidation error: %s", prefix, err)
        for warn in self.warnings:
            logger.warning("%sValidation warning: %s", prefix, warn)


# ==========================================================================
# Schema Validator
# ==========================================================================


class SchemaValidator:
    """
    Static validators for each data contract in the LangGraph workflow.
    All methods return SchemaValidationResult — they never raise.
    """

    # ==================================================================
    # 1. Pydantic model validators
    # ==================================================================

    @classmethod
    def validate_research_data(cls, data: Any) -> SchemaValidationResult:
        """
        Validate a research_data dict against the ResearchData Pydantic schema.
        Called after ResearchService.run() before storing in ContentState.
        """
        if not isinstance(data, dict):
            return SchemaValidationResult.failure(
                [f"research_data must be a dict, got {type(data).__name__}."]
            )

        try:
            model = ResearchData(**data)
            warnings: List[str] = []

            if model.total_documents == 0:
                warnings.append("research_data has zero documents — SEO scoring will degrade.")
            if not model.statistics:
                warnings.append("research_data has no statistics — writer prompts will lack data points.")
            if not model.citations:
                warnings.append("research_data has no citations — CitationService output will be empty.")

            return SchemaValidationResult.success(model.model_dump(), warnings)

        except ValidationError as exc:
            return SchemaValidationResult.failure(
                [f"ResearchData: {err['loc']} — {err['msg']}" for err in exc.errors()]
            )
        except Exception as exc:
            return SchemaValidationResult.failure(
                [f"ResearchData unexpected error: {exc}"]
            )

    @classmethod
    def validate_strategy(cls, data: Any) -> SchemaValidationResult:
        """
        Validate a strategy dict against the Strategy Pydantic schema.
        Called after StrategyAgent sets ContentState["strategy"].
        """
        if not isinstance(data, dict):
            return SchemaValidationResult.failure(
                [f"strategy must be a dict, got {type(data).__name__}."]
            )

        try:
            model = Strategy(**data)
            warnings: List[str] = []

            if not model.keywords:
                warnings.append("strategy has no primary keywords — SEO will be weak.")
            if not model.outline:
                warnings.append("strategy has no outline — WriterService will generate one via LLM.")
            if not model.cta:
                warnings.append("strategy has no CTA — review will flag cta_effectiveness as low.")
            if not model.seo.meta_title:
                warnings.append("strategy.seo has no meta_title.")

            return SchemaValidationResult.success(model.model_dump(), warnings)

        except ValidationError as exc:
            return SchemaValidationResult.failure(
                [f"Strategy: {err['loc']} — {err['msg']}" for err in exc.errors()]
            )
        except Exception as exc:
            return SchemaValidationResult.failure(
                [f"Strategy unexpected error: {exc}"]
            )

    @classmethod
    def validate_review(cls, data: Any) -> SchemaValidationResult:
        """
        Validate a review dict against the ReviewResult Pydantic schema.
        Called after ReviewService.run() before storing in ContentState.
        """
        if not isinstance(data, dict):
            return SchemaValidationResult.failure(
                [f"review must be a dict, got {type(data).__name__}."]
            )

        # Flatten dimension_scores if it came in as a plain dict
        dim_raw = data.get("dimension_scores", {})

        try:
            model = ReviewResult.from_service_dict(data)
            warnings: List[str] = []

            if model.score == 0:
                warnings.append("Review score is 0 — LLM evaluation may have failed entirely.")
            if model.needs_revision and not model.rewrite_instruction:
                warnings.append(
                    "Review status is FAIL but rewrite_instruction is empty — writer will have no guidance."
                )
            if model.status == "PASS" and model.score < _PASS_THRESHOLD:
                warnings.append(
                    f"Review forced PASS at score {model.score} (below threshold {_PASS_THRESHOLD}) — revision limit reached."
                )

            return SchemaValidationResult.success(model.to_state_dict(), warnings)

        except ValidationError as exc:
            return SchemaValidationResult.failure(
                [f"ReviewResult: {err['loc']} — {err['msg']}" for err in exc.errors()]
            )
        except Exception as exc:
            return SchemaValidationResult.failure(
                [f"ReviewResult unexpected error: {exc}"]
            )

    # ==================================================================
    # 2. ContentState field validators
    # ==================================================================

    @classmethod
    def validate_state_fields(
        cls,
        state: Dict[str, Any],
        required_fields: List[str],
    ) -> SchemaValidationResult:
        """
        Check that `required_fields` are present and non-empty in `state`.
        Used as the building block for all agent boundary validators.
        """
        errors: List[str] = []
        warnings: List[str] = []

        for field_name in required_fields:
            if field_name not in state:
                errors.append(f"ContentState missing required field: '{field_name}'.")
                continue

            value = state[field_name]

            # None is always invalid for required fields
            if value is None:
                errors.append(f"ContentState['{field_name}'] is None.")
                continue

            # Type-aware emptiness checks
            if isinstance(value, str) and not value.strip():
                errors.append(f"ContentState['{field_name}'] is an empty string.")
            elif isinstance(value, (list, dict)) and len(value) == 0:
                warnings.append(f"ContentState['{field_name}'] is empty (empty list or dict).")

        if errors:
            return SchemaValidationResult.failure(errors)
        return SchemaValidationResult.success(warnings=warnings)

    @classmethod
    def validate_enum_fields(cls, state: Dict[str, Any]) -> SchemaValidationResult:
        """
        Validate that enum-constrained ContentState fields hold legal values.
        Warnings only (not errors) — defaults are applied by the Manager.
        """
        warnings: List[str] = []

        content_type = str(state.get("content_type", "")).lower()
        if content_type and content_type not in _VALID_CONTENT_TYPES:
            warnings.append(
                f"content_type '{content_type}' is not in {_VALID_CONTENT_TYPES}."
            )

        platform = str(state.get("platform", "")).lower()
        if platform and platform not in _VALID_PLATFORMS:
            warnings.append(
                f"platform '{platform}' is not in {_VALID_PLATFORMS}."
            )

        objective = str(state.get("objective", "")).lower()
        if objective and objective not in _VALID_OBJECTIVES:
            warnings.append(
                f"objective '{objective}' is not in {_VALID_OBJECTIVES}."
            )

        workflow_status = str(state.get("workflow_status", ""))
        if workflow_status and workflow_status not in _VALID_WORKFLOW_STATUSES:
            warnings.append(
                f"workflow_status '{workflow_status}' is not in {_VALID_WORKFLOW_STATUSES}."
            )

        next_agent = str(state.get("next_agent", ""))
        if next_agent and next_agent not in _VALID_AGENTS:
            warnings.append(
                f"next_agent '{next_agent}' is not in {_VALID_AGENTS}."
            )

        return SchemaValidationResult.success(warnings=warnings)

    # ==================================================================
    # 3. Agent boundary validators
    # ==================================================================

    @classmethod
    def validate_for_research(cls, state: Dict[str, Any]) -> SchemaValidationResult:
        """
        Validate that ContentState is ready for the Research Agent.
        Called at the Manager → Research edge.
        Required: user_input, brand_context (with namespace).
        """
        errors: List[str] = []
        warnings: List[str] = []

        field_result = cls.validate_state_fields(
            state,
            required_fields=["user_input", "brand_context", "content_type", "platform"],
        )
        errors.extend(field_result.errors)
        warnings.extend(field_result.warnings)

        brand = state.get("brand_context", {})
        if isinstance(brand, dict):
            if not brand.get("namespace"):
                errors.append("brand_context missing 'namespace' — Pinecone KB search will fail.")
            if not brand.get("keyword_direction"):
                warnings.append("brand_context has no 'keyword_direction' — research queries will be less targeted.")
        else:
            errors.append(f"brand_context must be a dict, got {type(brand).__name__}.")

        enum_result = cls.validate_enum_fields(state)
        warnings.extend(enum_result.warnings)

        if errors:
            return SchemaValidationResult.failure(errors)
        return SchemaValidationResult.success(warnings=warnings)

    @classmethod
    def validate_for_strategy(cls, state: Dict[str, Any]) -> SchemaValidationResult:
        """
        Validate that ContentState is ready for the Strategy Agent.
        Called at the Research → Strategy edge.
        Required: research_data (with documents).
        """
        errors: List[str] = []
        warnings: List[str] = []

        field_result = cls.validate_state_fields(
            state,
            required_fields=["user_input", "brand_context", "research_data"],
        )
        errors.extend(field_result.errors)
        warnings.extend(field_result.warnings)

        # Validate research_data shape
        if "research_data" in state:
            research_result = cls.validate_research_data(state["research_data"])
            if not research_result.ok:
                errors.extend(research_result.errors)
            warnings.extend(research_result.warnings)

        if errors:
            return SchemaValidationResult.failure(errors)
        return SchemaValidationResult.success(warnings=warnings)

    @classmethod
    def validate_for_writer(cls, state: Dict[str, Any]) -> SchemaValidationResult:
        """
        Validate that ContentState is ready for the Writer Agent.
        Called at the Strategy → Writer edge.
        Required: strategy (with keywords), research_data.
        """
        errors: List[str] = []
        warnings: List[str] = []

        field_result = cls.validate_state_fields(
            state,
            required_fields=["user_input", "brand_context", "research_data", "strategy"],
        )
        errors.extend(field_result.errors)
        warnings.extend(field_result.warnings)

        if "strategy" in state:
            strategy_result = cls.validate_strategy(state["strategy"])
            if not strategy_result.ok:
                errors.extend(strategy_result.errors)
            warnings.extend(strategy_result.warnings)

        if errors:
            return SchemaValidationResult.failure(errors)
        return SchemaValidationResult.success(warnings=warnings)

    @classmethod
    def validate_for_review(cls, state: Dict[str, Any]) -> SchemaValidationResult:
        """
        Validate that ContentState is ready for the Review Agent.
        Called at the Writer → Review edge.
        Required: draft (non-empty), strategy, brand_context.
        """
        errors: List[str] = []
        warnings: List[str] = []

        field_result = cls.validate_state_fields(
            state,
            required_fields=["draft", "strategy", "brand_context"],
        )
        errors.extend(field_result.errors)
        warnings.extend(field_result.warnings)

        # Draft-specific checks
        draft = state.get("draft", "")
        if isinstance(draft, str) and draft.strip():
            word_count = len(draft.split())
            if word_count < 100:
                warnings.append(
                    f"Draft is very short ({word_count} words) — review pre-checks may fail."
                )
            if not draft.startswith("#"):
                warnings.append(
                    "Draft does not start with a Markdown H1 heading — "
                    "MetadataService title extraction will fall back to strategy.title."
                )

        if errors:
            return SchemaValidationResult.failure(errors)
        return SchemaValidationResult.success(warnings=warnings)

    @classmethod
    def validate_after_review(cls, state: Dict[str, Any]) -> SchemaValidationResult:
        """
        Validate that ContentState is complete after the Review Agent.
        Called before routing to END or back to Writer.
        Required: review (with score, status, needs_revision).
        """
        errors: List[str] = []
        warnings: List[str] = []

        field_result = cls.validate_state_fields(
            state,
            required_fields=["review", "draft", "final_output"],
        )
        errors.extend(field_result.errors)
        warnings.extend(field_result.warnings)

        if "review" in state and isinstance(state["review"], dict):
            review_result = cls.validate_review(state["review"])
            if not review_result.ok:
                errors.extend(review_result.errors)
            warnings.extend(review_result.warnings)

        # Revision count sanity check
        revision_count = state.get("revision_count", 0)
        max_revisions = state.get("max_revision_count", 3)
        if revision_count > max_revisions:
            warnings.append(
                f"revision_count ({revision_count}) exceeds max_revision_count ({max_revisions}) — "
                "Review Agent should have forced PASS."
            )

        if errors:
            return SchemaValidationResult.failure(errors)
        return SchemaValidationResult.success(warnings=warnings)

    @classmethod
    def validate_final_output(cls, state: Dict[str, Any]) -> SchemaValidationResult:
        """
        Validate that final_output is complete and well-formed before
        returning the result to the caller.
        """
        errors: List[str] = []
        warnings: List[str] = []

        final = state.get("final_output", {})
        if not isinstance(final, dict) or not final:
            errors.append("final_output is missing or empty.")
            return SchemaValidationResult.failure(errors)

        # Check core output fields
        required_output_keys = ["content", "metadata", "seo"]
        for key in required_output_keys:
            if key not in final:
                errors.append(f"final_output missing required key: '{key}'.")

        # Warn on empty optional fields
        if not final.get("hashtags"):
            warnings.append("final_output has no hashtags.")
        if not final.get("citations"):
            warnings.append("final_output has no citations.")

        # Confirm workflow completed successfully
        if state.get("workflow_status") != "COMPLETED":
            warnings.append(
                f"workflow_status is '{state.get('workflow_status')}' — expected 'COMPLETED'."
            )

        if errors:
            return SchemaValidationResult.failure(errors)
        return SchemaValidationResult.success(warnings=warnings)

    # ==================================================================
    # 4. Convenience: validate full state at once
    # ==================================================================

    @classmethod
    def validate_complete_state(cls, state: Dict[str, Any]) -> SchemaValidationResult:
        """
        Run all boundary validators in sequence and aggregate results.
        Useful for testing and debugging a full workflow run.
        Returns combined errors and warnings from all checks.
        """
        all_errors: List[str] = []
        all_warnings: List[str] = []

        checks = [
            ("for_research", cls.validate_for_research),
            ("for_strategy", cls.validate_for_strategy),
            ("for_writer", cls.validate_for_writer),
            ("for_review", cls.validate_for_review),
            ("after_review", cls.validate_after_review),
            ("final_output", cls.validate_final_output),
            ("enum_fields", cls.validate_enum_fields),
        ]

        for name, check_fn in checks:
            result = check_fn(state)
            if not result.ok:
                all_errors.extend([f"[{name}] {e}" for e in result.errors])
            all_warnings.extend([f"[{name}] {w}" for w in result.warnings])

        if all_errors:
            return SchemaValidationResult.failure(all_errors)
        return SchemaValidationResult.success(warnings=all_warnings)
