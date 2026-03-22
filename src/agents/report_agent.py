"""Report agent — cross-candidate synthesis into a recruiter briefing.

Uses Claude Sonnet (claude-sonnet-4-5) via the Anthropic API.
Produces a markdown recruiter briefing with cross-candidate analysis.

Takes all scored CandidateScore results together and produces a
markdown summary with recommendations, skill gap analysis, tailored
interview questions, and red flags.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.core.models import CandidateScore, JobDescription

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class ReportAgent:
    """Synthesis agent that generates a recruiter briefing.

    Operates on all CandidateScore results together to produce
    cross-candidate insights that a per-resume scorer cannot provide.
    """

    def __init__(self, model: BaseChatModel) -> None:
        self._model = model
        self._template = (PROMPTS_DIR / "report.txt").read_text()

    def run(
        self,
        results: list[CandidateScore],
        jd: JobDescription,
        output_path: str = "src/results/recruiter_summary.md",
    ) -> str:
        """Generate recruiter_summary.md from all scored candidates.

        Args:
            results:     Scored candidates (already sorted by score).
            jd:          The structured JobDescription.
            output_path: Where to write the markdown file.

        Returns:
            The path to the written file.
        """
        prompt = self._build_prompt(results=results, jd=jd)
        response = self._model.invoke([HumanMessage(content=prompt)])
        summary = response.content.strip()

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(summary, encoding="utf-8")

        logger.info("[Report]      recruiter_summary.md written to %s", output_path)
        return output_path

    def _build_prompt(
        self,
        results: list[CandidateScore],
        jd: JobDescription,
    ) -> str:
        """Fill prompts/report.txt with candidate data."""
        candidates_summary = self._format_candidates(results)
        return self._template.format(
            job_title=jd.job_title,
            candidate_count=len(results),
            candidates_summary=candidates_summary,
        )

    def _format_candidates(self, results: list[CandidateScore]) -> str:
        """Format all candidates into a text block for the prompt.

        Includes rank, filename, score, fit label, explanation,
        and top 3 requirement scores with evidence.
        """
        lines: list[str] = []
        for i, r in enumerate(results, start=1):
            lines.append(f"Candidate {i}: {r.resume.filename}")
            lines.append(f"  Score: {r.overall_score}/100 ({r.fit_label})")
            lines.append(f"  Summary: {r.explanation}")
            lines.append("  Top requirement scores:")
            top_reqs = sorted(
                r.requirement_scores,
                key=lambda x: x.score,
                reverse=True,
            )[:3]
            for req in top_reqs:
                lines.append(f"    - {req.requirement}: {req.score}/100")
                lines.append(f"      Evidence: {req.evidence}")
            lines.append("")
        return "\n".join(lines)
