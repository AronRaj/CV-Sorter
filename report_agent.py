"""Report agent — cross-candidate synthesis into a recruiter briefing.

**Role in the multi-agent pipeline**

`Supervisor` invokes this agent **after** `ScorerAgent` has produced a sorted
`list[CandidateScore]`. Unlike scoring, which is per-resume, the report stage
must see **all candidates at once** to rank narratives, contrast strengths,
surface common gaps, and propose interview angles that only make sense in
relative terms. That is why this agent exists as a separate step rather than
extending the per-file scorer prompt.

**Why Claude Sonnet (same tier as scoring)**

The briefing is user-facing markdown: tone, structure, and factual grounding
matter. Reusing the same API model family as `ScorerAgent` keeps stylistic
consistency and leverages long-context packing of multiple scorecards. The
shortlist model is intentionally **not** used here — local small models are prone
to shallow comparisons when many candidates are in one prompt.

**Data flow**

Input: `results` (already sorted by `overall_score`), `jd` for role context.
The agent formats a condensed text block per candidate (top requirements +
evidence) to control token use, fills `prompts/report.txt`, invokes the model
once, writes markdown to `output_path`, and returns that path for the
supervisor's result dict.

**Failure handling**

There is no retry loop in this module: a single `invoke` is assumed sufficient
because the input is structured text, not fragile JSON. Operational failures
(billing, network) propagate to the caller.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from models import CandidateScore, JobDescription

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class ReportAgent:
    """Synthesis agent: one LLM call over all `CandidateScore` rows → markdown file.

    Complements `ScorerAgent` by operating on the **aggregate** view. Prompt
    assembly is split into `_build_prompt` and `_format_candidates` so tests can
    inspect formatting without mocking the model.
    """

    def __init__(self, model: BaseChatModel) -> None:
        """Load the report template and store the shared high-quality chat model.

        Args:
            model: Same Claude instance family as scoring — intentional for
                consistent reasoning and prose.

        Side effects:
            Reads ``prompts/report.txt`` at construction.
        """
        self._model = model
        self._template = (PROMPTS_DIR / "report.txt").read_text()

    def run(
        self,
        results: list[CandidateScore],
        jd: JobDescription,
        output_path: str = "results/recruiter_summary.md",
    ) -> str:
        """Generate the recruiter markdown summary and persist it to disk.

        The model output is written verbatim (after strip) — the template is
        responsible for asking for markdown sections the product expects.

        Args:
            results: Scored candidates; typically pre-sorted by score so rank in
                the prompt matches recruiter expectations.
            jd: Job description for role title and context in the template.
            output_path: Filesystem path for the `.md` artifact.

        Returns:
            The ``output_path`` string (for logging and supervisor return dict).

        Side effects:
            Creates parent directories if needed; overwrites ``output_path``;
            one LLM `invoke`.
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
        """Inject job metadata and formatted candidate summaries into `report.txt`.

        Args:
            results: All candidates to include in the cross-candidate analysis.
            jd: Supplies `job_title` and anchors the report to the correct role.

        Returns:
            Final prompt string passed to the model.

        Side effects:
            None.
        """
        candidates_summary = self._format_candidates(results)
        return self._template.format(
            job_title=jd.job_title,
            candidate_count=len(results),
            candidates_summary=candidates_summary,
        )

    def _format_candidates(self, results: list[CandidateScore]) -> str:
        """Build the large middle section of the report prompt from score objects.

        We deliberately **truncate** per-candidate detail to the top three
        requirement scores by points — the full rubric can be enormous and would
        dominate context. The report is meant to synthesize, not duplicate every
        line item from JSON output.

        Args:
            results: Ordered list of `CandidateScore` (order preserved in output).

        Returns:
            Multi-line string: for each candidate, rank, file, score, summary,
            and top requirements with evidence lines.

        Side effects:
            None.
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
