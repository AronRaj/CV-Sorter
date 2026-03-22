"""All data structures for the CV Sorter project.

Single source of truth for every data shape used across the system.
No logic, no imports from other src/ files — dataclasses only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Resume:
    """A parsed candidate resume extracted from a PDF file."""

    filename: str
    raw_text: str
    name: str | None
    skills: list[str]
    experience_years: int | None
    education: list[str]


@dataclass
class JobDescription:
    """Structured representation of a job description extracted via LLM."""

    raw_text: str
    job_title: str
    required_skills: list[str]
    nice_to_have_skills: list[str]
    min_experience_years: int | None
    responsibilities: list[str]


@dataclass
class RequirementScore:
    """Score for a single job requirement against a candidate resume."""

    requirement: str
    score: int  # 0–100
    evidence: str  # LLM explanation


@dataclass
class CandidateScore:
    """Overall scoring result for one candidate against a job description."""

    resume: Resume
    overall_score: int  # 0–100
    fit_label: str  # "Strong match" | "Good match" | "Partial match" | "Weak match"
    explanation: str
    requirement_scores: list[RequirementScore]
