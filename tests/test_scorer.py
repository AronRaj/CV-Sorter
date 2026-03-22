"""Unit tests for src/scorer.py.

Uses mock chat models that return hardcoded strings — no real API calls.
"""

import json

import pytest

from src.core.models import JobDescription, Resume
from src.core.scorer_engine import Scorer

VALID_JSON_RESPONSE = """
{
  "overall_score": 82,
  "fit_label": "Strong match",
  "explanation": "The candidate has strong Python experience matching the JD.",
  "requirement_scores": [
    {
      "requirement": "Python 5+ years",
      "score": 90,
      "evidence": "CV section 2 states 6 years of Python development."
    },
    {
      "requirement": "PostgreSQL experience",
      "score": 75,
      "evidence": "Listed PostgreSQL under core skills with 3 years noted."
    }
  ]
}
"""


class _Msg:
    """Minimal AIMessage stand-in with a .content attribute."""

    def __init__(self, content: str) -> None:
        self.content = content


class MockChatModel:
    """Test double for BaseChatModel. Returns a hardcoded response."""

    def __init__(self, response: str) -> None:
        self._response = response

    def invoke(self, messages: list) -> _Msg:
        """Return an object with a .content attribute."""
        return _Msg(self._response)


class CountingMockClient:
    """Test double that returns invalid JSON on first call, valid on second."""

    def __init__(self) -> None:
        self.call_count = 0

    def invoke(self, messages: list) -> _Msg:
        """Return invalid JSON first, then valid JSON."""
        self.call_count += 1
        if self.call_count == 1:
            return _Msg("this is not json at all")
        return _Msg(VALID_JSON_RESPONSE)


def _make_resume() -> Resume:
    return Resume(
        filename="test_candidate.pdf",
        raw_text="dummy resume text",
        name=None,
        skills=[],
        experience_years=None,
        education=[],
    )


def _make_jd() -> JobDescription:
    return JobDescription(
        raw_text="dummy jd text",
        job_title="Engineer",
        required_skills=["Python"],
        nice_to_have_skills=["Docker"],
        min_experience_years=3,
        responsibilities=["Build APIs"],
    )


def test_score_returns_candidate_score() -> None:
    """Verify scorer parses valid JSON into a correct CandidateScore."""
    scorer = Scorer(model=MockChatModel(VALID_JSON_RESPONSE))
    result = scorer.score(resume=_make_resume(), jd=_make_jd())

    assert result.overall_score == 82
    assert result.fit_label == "Strong match"
    assert len(result.requirement_scores) == 2
    assert result.requirement_scores[0].score == 90
    assert result.resume.filename == "test_candidate.pdf"


def test_parse_response_strips_code_fences() -> None:
    """Verify scorer handles JSON wrapped in markdown code fences."""
    fenced = "```json\n" + VALID_JSON_RESPONSE + "\n```"
    scorer = Scorer(model=MockChatModel(fenced))
    result = scorer.score(resume=_make_resume(), jd=_make_jd())

    assert result.overall_score == 82


def test_invalid_json_triggers_retry() -> None:
    """Verify scorer retries once when the first response is not valid JSON."""
    client = CountingMockClient()
    scorer = Scorer(model=client)
    result = scorer.score(resume=_make_resume(), jd=_make_jd())

    assert result.overall_score == 82
    assert client.call_count == 2


def test_two_invalid_responses_raises_runtime_error() -> None:
    """Verify scorer raises RuntimeError after two consecutive parse failures."""
    scorer = Scorer(model=MockChatModel("not json"))

    with pytest.raises(RuntimeError, match="test_candidate.pdf"):
        scorer.score(resume=_make_resume(), jd=_make_jd())


def test_fit_label_values() -> None:
    """Verify scorer correctly passes through all valid fit_label values."""
    valid_labels = ["Strong match", "Good match", "Partial match", "Weak match"]

    for label in valid_labels:
        data = json.loads(VALID_JSON_RESPONSE)
        data["fit_label"] = label
        response = json.dumps(data)

        scorer = Scorer(model=MockChatModel(response))
        result = scorer.score(resume=_make_resume(), jd=_make_jd())

        assert result.fit_label in valid_labels
