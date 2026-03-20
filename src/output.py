"""Output module for ranked scoring results.

Serialises ranked CandidateScore results to a JSON file and provides
a Rich terminal table for --verbose display.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.models import CandidateScore, JobDescription

logger = logging.getLogger(__name__)

FIT_LABEL_COLOURS = {
    "Strong match": "green",
    "Good match": "blue",
    "Partial match": "yellow",
    "Weak match": "red",
}


class OutputWriter:
    """Writes ranked scoring results to JSON and prints Rich terminal tables."""

    def write(
        self,
        ranked: list[CandidateScore],
        jd: JobDescription,
        provider: str,
        model: str,
        output_path: str,
    ) -> None:
        """Serialise ranked results to a JSON file.

        Creates the parent directory if it does not exist.

        Args:
            ranked: CandidateScore list sorted by overall_score descending.
            jd: The JobDescription used for scoring.
            provider: LLM provider name (e.g. "groq", "claude").
            model: Model string used (e.g. "llama-3.3-70b-versatile").
            output_path: File path for the output JSON.
        """
        output_data = {
            "job_title": jd.job_title,
            "provider": provider,
            "model": model,
            "ranked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "total_candidates": len(ranked),
            "candidates": [
                self._build_candidate_entry(rank=i + 1, result=result)
                for i, result in enumerate(ranked)
            ],
        }

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info("Results written to %s", output_path)

    def print_table(self, ranked: list[CandidateScore]) -> None:
        """Print a Rich terminal table of ranked results.

        Displays rank, candidate filename, score, and colour-coded fit label.
        Called by main.py only when --verbose is set. Does not write to disk.

        Args:
            ranked: CandidateScore list sorted by overall_score descending.
        """
        from rich.console import Console
        from rich.table import Table

        table = Table(title="Candidate Rankings")
        table.add_column("Rank", justify="center")
        table.add_column("Candidate", justify="left")
        table.add_column("Score", justify="right")
        table.add_column("Fit", justify="left")

        for i, result in enumerate(ranked, start=1):
            colour = FIT_LABEL_COLOURS.get(result.fit_label, "white")
            table.add_row(
                str(i),
                result.resume.filename,
                str(result.overall_score),
                f"[{colour}]{result.fit_label}[/{colour}]",
            )

        Console().print(table)

    def write_csv(
        self,
        ranked: list[CandidateScore],
        output_path: str,
    ) -> None:
        """Write ranked candidates to a CSV file.

        One row per candidate. Columns: Rank, Filename, Overall Score,
        Fit Label, Explanation, plus two columns per requirement
        ("<requirement> Score" and "<requirement> Evidence").

        Creates the parent directory if it does not exist.
        Uses utf-8-sig encoding so Excel opens it correctly on all platforms.

        Args:
            ranked: CandidateScore list sorted by overall_score descending.
            output_path: File path for the output CSV.
        """
        requirements = list(
            dict.fromkeys(
                rs.requirement
                for result in ranked
                for rs in result.requirement_scores
            )
        )

        header = ["Rank", "Filename", "Overall Score", "Fit Label", "Explanation"]
        for req in requirements:
            header.append(f"{req} Score")
            header.append(f"{req} Evidence")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for i, result in enumerate(ranked, start=1):
                scores_by_req = {
                    rs.requirement: rs for rs in result.requirement_scores
                }
                row: list[str | int] = [
                    i,
                    result.resume.filename,
                    result.overall_score,
                    result.fit_label,
                    result.explanation,
                ]
                for req in requirements:
                    rs = scores_by_req.get(req)
                    if rs:
                        row.append(rs.score)
                        row.append(rs.evidence)
                    else:
                        row.append("")
                        row.append("")
                writer.writerow(row)

        logger.info("CSV written to %s", output_path)

    def _build_candidate_entry(self, rank: int, result: CandidateScore) -> dict:
        """Build a single candidate dict for the JSON output.

        Args:
            rank: 1-based rank position.
            result: The CandidateScore for this candidate.

        Returns:
            A dict matching the output JSON schema.
        """
        return {
            "rank": rank,
            "filename": result.resume.filename,
            "overall_score": result.overall_score,
            "fit_label": result.fit_label,
            "explanation": result.explanation,
            "requirement_scores": [
                {
                    "requirement": rs.requirement,
                    "score": rs.score,
                    "evidence": rs.evidence,
                }
                for rs in result.requirement_scores
            ],
        }
