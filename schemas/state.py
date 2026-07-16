"""
Shared LangGraph workflow state.

Rules
-----
- Every agent reads from this TypedDict.
- Every agent ONLY writes to the fields it owns.
- Keep field names stable — renaming breaks graph edges.
- Use the Pydantic schemas (research_schema, strategy_schema, review_schema)
  to validate data before storing it here.

Field ownership
---------------
  Manager   → brand_context, primary_topic, user_constraints, safety,
               workflow_status (INIT → RUNNING | BLOCKED),
               current_agent, next_agent
  Research  → research_data, retrieved_documents, sources
  Strategy  → strategy, seo, hashtags
  Writer    → draft, metadata, formatted_output, final_output
  Review    → review, revision_count, workflow_status (→ COMPLETED | BLOCKED)
"""

from typing import Any, Dict, List, TypedDict


class ContentState(TypedDict):

    # ==========================================================================
    # Identity
    # ==========================================================================

    request_id: str
    """UUID generated at workflow entry — used as the MongoDB workflow-run key."""

    session_id: str
    """
    Conversation session identifier for ConversationMemory.
    Allows multiple workflow runs to share the same memory session
    (e.g. a user refining the same article across several requests).
    """

    # ==========================================================================
    # Request
    # ==========================================================================

    user_input: str
    """Raw text entered by the user (topic, question, or brief)."""

    brand: str | None
    """Brand selected by the user. None means auto-detect."""

    content_type: str
    """article | blog | linkedin | email | carousel"""

    platform: str
    """website | linkedin | email | x"""

    objective: str
    """seo | engagement | authority | leads"""

    language: str
    """English | Hindi  (default: English)"""

    additional_instructions: str
    """Optional free-text modifier appended to writer prompts."""

    primary_topic: str
    """
    Locked topic summary from the Manager safety gate.
    Downstream agents must follow this and must not invert meaning/roles.
    """

    user_constraints: Dict[str, Any]
    """
    Parsed user constraints from Manager. Shape:
    {
        "target_word_count": int | None,
        "word_count_flexible": bool,
        "raw_length_mentions": List[int],
    }
    """

    safety: Dict[str, Any]
    """
    Policy decision from SafetyService. Shape:
    {
        "allowed": bool,
        "blocked": bool,
        "category": str,
        "reason": str,
        "message": str,
        "defensive_allow": bool,
    }
    """

    # ==========================================================================
    # Business Context  (Manager Agent + BusinessContextService)
    # ==========================================================================

    brand_context: Dict[str, Any]
    """
    Resolved brand configuration. Shape:
    {
        "brand"            : str,
        "namespace"        : str,       # Pinecone KB namespace
        "display_name"     : str,
        "tone"             : str,
        "reader_segment"   : List[str],
        "pain_points"      : List[str],
        "keyword_direction": List[str],
        "cta"              : str,
    }
    """

    # ==========================================================================
    # Research Agent
    # ==========================================================================

    research_data: Dict[str, Any]
    """
    Full research package from ResearchService (validated by ResearchData schema).
    Keys: documents, total_documents, sources, statistics, citations.
    """

    retrieved_documents: List[Dict[str, Any]]
    """
    Flat list of ResearchDocument dicts — shortcut used by WriterService
    and SEOService without unpacking research_data.
    """

    sources: List[Dict[str, Any]]
    """
    Flat list of ResearchSource dicts — used by CitationService
    and final_output assembly.
    """

    # ==========================================================================
    # Strategy Agent
    # ==========================================================================

    strategy: Dict[str, Any]
    """
    Complete strategy dict (validated by Strategy schema). Shape:
    {
        "title"              : str,
        "content_angle"      : str,
        "audience"           : List[str],
        "tone"               : str,
        "outline"            : List[{heading, heading_level, brief, keywords}],
        "cta"                : str,
        "content_type"       : str,
        "platform"           : str,
        "language"           : str,
        "keywords"           : List[str],   # primary keywords
        "secondary_keywords" : List[str],
        "pain_points"        : List[str],
        "seo"                : SEOBlueprint dict,
        "hashtags"           : List[str],
        "citations"          : List[str],
        "rewrite_instruction": str,         # injected by Review Agent on FAIL
    }
    """

    seo: Dict[str, Any]
    """
    SEOBlueprint dict — mirrors strategy["seo"], surfaced at the top level
    so the Formatter and JSONBuilder can access it without unpacking strategy.
    Keys: primary_keywords, secondary_keywords, keyword_scores,
          search_intent, meta_title, meta_description, slug.
    """

    hashtags: List[str]
    """Platform-optimised hashtags from HashtagService."""

    # ==========================================================================
    # Writer Agent
    # ==========================================================================

    draft: str
    """Full Markdown content draft produced by WriterService."""

    metadata: Dict[str, Any]
    """
    Content metadata from MetadataService. Shape:
    {
        "word_count"     : int,
        "read_time_min"  : int,
        "language"       : str,
        "content_type"   : str,
        "platform"       : str,
        "primary_keyword": str,
        "slug"           : str,
        "meta_title"     : str,
        "meta_description": str,
    }
    """

    formatted_output: Dict[str, Any]
    """Platform-specific formatted content from Formatter."""

    final_output: Dict[str, Any]
    """
    Complete structured response from JSONBuilder. Shape:
    {
        "content"   : str | Dict,
        "metadata"  : Dict,
        "seo"       : Dict,
        "hashtags"  : List[str],
        "citations" : List[str],
    }
    """

    # ==========================================================================
    # Review Agent
    # ==========================================================================

    review: Dict[str, Any]
    """
    Review decision dict (validated by ReviewResult schema). Shape:
    {
        "score"              : int,        # 0–100 weighted composite
        "status"             : str,        # "PASS" | "FAIL"
        "needs_revision"     : bool,
        "feedback"           : List[str],  # what the content does well
        "issues"             : List[str],  # problems found
        "rewrite_instruction": str,        # writer brief (empty on PASS)
        "dimension_scores"   : {
            "content_quality"  : int,
            "seo_compliance"   : int,
            "brand_alignment"  : int,
            "structure"        : int,
            "cta_effectiveness": int,
        },
        "revision_number"    : int,
    }
    """

    # ==========================================================================
    # Workflow Management
    # ==========================================================================

    revision_count: int
    """
    How many rewrite cycles have completed.
    Incremented by the Review Agent on each FAIL → writer routing.
    """

    max_revision_count: int
    """
    Hard cap on rewrites (default: settings.MAX_REVIEW_ITERATIONS = 0).
    When revision_count >= max_revision_count, Review Agent forces PASS.
    """

    current_agent: str
    """Name of the agent that last updated state (manager | research | strategy | writer | review)."""

    next_agent: str
    """Name of the agent that will run next (research | strategy | writer | review | end)."""

    workflow_status: str
    """
    INIT       — state created, not yet entered graph
    RUNNING    — graph is executing
    COMPLETED  — review passed, final_output is ready
    BLOCKED    — safety/policy refused; no content may be returned
    FAILED     — unrecoverable error occurred
    """

    # ==========================================================================
    # Error Handling
    # ==========================================================================

    errors: List[str]
    """
    Accumulates non-fatal error messages from any agent or service.
    Fatal errors raise exceptions and halt the graph immediately.
    """
