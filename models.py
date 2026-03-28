"""All data structures for the CV Sorter project.

This module is the **contract** between parsing, scoring, agents, and output:
every layer exchanges these datatypes so evaluators can trace data flow from
PDF → structured resume → scores → reports without hunting through ad-hoc dicts.

Design constraints:

* **No business logic** here — only immutable-shaped records (dataclasses).
* **No imports from other application packages** — avoids circular dependencies
  and keeps this file safe as a lowest-level dependency.

Field types favor explicit lists and optional fields (``None``) where the
pipeline could not extract a value, so callers can branch or default explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Parsed inputs (resume + job description)
# ---------------------------------------------------------------------------


@dataclass
class Resume:
    """A candidate resume after PDF/text extraction and light structuring.

    Populated by the parser layer from a single file; ``raw_text`` preserves
    the full extraction for LLM context while structured fields support
    deterministic checks and UI summaries.

    Attributes:
        filename: Original basename or path segment used to identify the CV
            in logs and multi-file batches.
        raw_text: Full text extracted from the document; primary input for
            LLM scoring when semantic nuance matters more than structured fields.
        name: Candidate name if detected; ``None`` when parsing could not
            isolate it (downstream should not assume presence).
        skills: Normalized skill tokens or phrases as produced by the parser
            (ordering may reflect document order, not importance).
        experience_years: Total years of experience if inferred as a single
            integer; ``None`` when not extracted — scoring may fall back to text.
        education: Human-readable education lines or degrees; list may be empty.
    """

    filename: str
    raw_text: str
    name: str | None
    skills: list[str]
    experience_years: int | None
    education: list[str]


@dataclass
class JobDescription:
    """Structured job description produced from raw text (often via LLM).

    Separates **must-have** vs. **nice-to-have** skills so the scorer can
    weight requirements correctly instead of treating every bullet equally.

    Attributes:
        raw_text: Original JD text for traceability and optional re-parsing.
        job_title: Normalized title string for display and report headers.
        required_skills: Hard requirements; typically drive mandatory scoring dimensions.
        nice_to_have_skills: Secondary skills; may contribute to overall fit
            without failing candidates who lack them.
        min_experience_years: Minimum years if stated; ``None`` if the JD does
            not specify a numeric bar (scoring should not invent a threshold).
        responsibilities: Bullet-style responsibilities for qualitative matching.
    """

    raw_text: str
    job_title: str
    required_skills: list[str]
    nice_to_have_skills: list[str]
    min_experience_years: int | None
    responsibilities: list[str]


# ---------------------------------------------------------------------------
# Scoring artifacts (per-requirement and per-candidate)
# ---------------------------------------------------------------------------


@dataclass
class RequirementScore:
    """Evidence-backed score for one JD requirement against one resume.

    Storing requirement text alongside the score keeps reports self-contained
    when the JD is long or requirements are LLM-paraphrased (no fragile indices).

    Attributes:
        requirement: The requirement string as evaluated (may mirror JD text).
        score: Integer 0–100 indicating match strength for this requirement;
            interpretation is defined by the scorer prompt/engine, not this type.
        evidence: Short natural-language justification from the LLM or rules,
            intended for recruiter-facing transparency.
    """

    requirement: str
    score: int  # 0–100
    evidence: str  # LLM explanation


@dataclass
class CandidateScore:
    """Aggregate scoring outcome for one candidate versus one job description.

    ``fit_label`` is a coarse band derived from ``overall_score`` (or policy)
    so UIs and exports can show a quick verdict without re-deriving thresholds
    in every consumer.

    Attributes:
        resume: The :class:`Resume` that was scored; links structured CV data
            to the result for sorting and reporting.
        overall_score: Integer 0–100 composite across requirements; semantics
            match the project's scoring rubric.
        fit_label: Human-readable bucket such as ``"Strong match"``,
            ``"Good match"``, ``"Partial match"``, or ``"Weak match"`` —
            exact strings are part of the product contract for dashboards/docs.
        explanation: High-level narrative summarizing why the score was assigned.
        requirement_scores: Ordered (or requirement-ordered) breakdown used for
            drill-down tables and audit trails.
    """

    resume: Resume
    overall_score: int  # 0–100
    fit_label: str  # "Strong match" | "Good match" | "Partial match" | "Weak match"
    explanation: str
    requirement_scores: list[RequirementScore]
