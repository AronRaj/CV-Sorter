"""Unit tests for src/models.py dataclasses.

Pure construction and field-assertion tests — no file I/O, no LLM calls.
"""

from src.models import CandidateScore, JobDescription, RequirementScore, Resume


def test_resume_defaults() -> None:
    """Verify Resume stores optional fields as None/empty when not populated."""
    resume = Resume(
        filename="test.pdf",
        raw_text="some text",
        name=None,
        skills=[],
        experience_years=None,
        education=[],
    )
    assert resume.filename == "test.pdf"
    assert resume.raw_text == "some text"
    assert resume.name is None
    assert resume.skills == []
    assert resume.experience_years is None
    assert resume.education == []


def test_job_description_fields() -> None:
    """Verify JobDescription stores all fields correctly when fully populated."""
    jd = JobDescription(
        raw_text="We need an engineer.",
        job_title="Backend Engineer",
        required_skills=["Python", "SQL"],
        nice_to_have_skills=["Docker"],
        min_experience_years=3,
        responsibilities=["Build APIs", "Review code"],
    )
    assert jd.job_title == "Backend Engineer"
    assert jd.required_skills == ["Python", "SQL"]
    assert jd.nice_to_have_skills == ["Docker"]
    assert jd.min_experience_years == 3
    assert jd.responsibilities == ["Build APIs", "Review code"]


def test_requirement_score_fields() -> None:
    """Verify RequirementScore stores requirement, score, and evidence."""
    rs = RequirementScore(
        requirement="Python 5+ years",
        score=90,
        evidence="Candidate has 7 years of Python experience.",
    )
    assert rs.requirement == "Python 5+ years"
    assert rs.score == 90
    assert rs.evidence == "Candidate has 7 years of Python experience."


def test_candidate_score_fields() -> None:
    """Verify CandidateScore nests Resume and RequirementScore correctly."""
    resume = Resume(
        filename="alice.pdf",
        raw_text="Alice's resume",
        name="Alice",
        skills=["Python"],
        experience_years=5,
        education=["MSc CS"],
    )
    rs1 = RequirementScore(requirement="Python", score=95, evidence="Strong")
    rs2 = RequirementScore(requirement="SQL", score=70, evidence="Moderate")

    cs = CandidateScore(
        resume=resume,
        overall_score=82,
        fit_label="Strong match",
        explanation="Good overall fit.",
        requirement_scores=[rs1, rs2],
    )
    assert cs.overall_score == 82
    assert cs.fit_label == "Strong match"
    assert len(cs.requirement_scores) == 2
    assert cs.requirement_scores[0].requirement == "Python"


def test_score_range_is_integer() -> None:
    """Verify overall_score is stored as an integer."""
    resume = Resume(
        filename="bob.pdf",
        raw_text="Bob's resume",
        name=None,
        skills=[],
        experience_years=None,
        education=[],
    )
    cs = CandidateScore(
        resume=resume,
        overall_score=75,
        fit_label="Good match",
        explanation="Decent fit.",
        requirement_scores=[],
    )
    assert isinstance(cs.overall_score, int)
