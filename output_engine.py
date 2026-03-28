"""Output module for ranked scoring results.

Turns in-memory ``CandidateScore`` lists (already sorted by the caller) into
**durable artifacts**: JSON for programmatic consumption, optional CSV for
spreadsheets, and optional **Rich** tables for human-readable terminal output.

This module intentionally stays free of scoring logic—it only serialises and
formats data the rest of the pipeline produced.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from models import CandidateScore, JobDescription

logger = logging.getLogger(__name__)

# Rich markup colour names keyed by the exact fit_label strings the scorer emits.
# Unknown labels fall back to white in ``print_table`` so new labels do not crash the UI.
FIT_LABEL_COLOURS = {
    "Strong match": "green",
    "Good match": "blue",
    "Partial match": "yellow",
    "Weak match": "red",
}


class OutputWriter:
    """Serialises ranked results to JSON/CSV and prints optional Rich tables.

    All public methods assume ``ranked`` is **pre-sorted** (typically by
    ``overall_score`` descending); this class does not re-rank. File-writing
    methods create parent directories as needed so callers can pass nested paths
    without extra setup.
    """

    def write(
        self,
        ranked: list[CandidateScore],
        jd: JobDescription,
        provider: str,
        model: str,
        output_path: str,
    ) -> None:
        """Write a single JSON document capturing the run metadata and ranked candidates.

        The schema is stable for downstream tools: job context, provenance
        (provider/model), UTC timestamp, counts, and per-candidate detail
        including per-requirement scores. ``ranked_at`` uses ISO-8601 with a
        ``Z`` suffix so consumers do not need to parse ``+00:00`` variants.

        Args:
            ranked: Ordered list of ``CandidateScore`` (best first).
            jd: Job description whose title is stored for report context.
            provider: Logical provider or pipeline name (e.g. multi-agent vs single model).
            model: Model identifier string for reproducibility and audits.
            output_path: Destination file path (UTF-8 JSON).

        Returns:
            None

        Raises:
            OSError: If the path cannot be written (permissions, disk full, etc.).

        Side effects:
            Creates parent directories; overwrites ``output_path`` if it exists;
            logs INFO on success.
        """
        # Single payload keeps JSON self-describing for graders and external tools.
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
            # ensure_ascii=False preserves names and explanations in non-Latin scripts.
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info("Results written to %s", output_path)

    def print_table(self, ranked: list[CandidateScore]) -> None:
        """Render a colour-coded ranking table to the terminal using Rich.

        Importing Rich here (lazy import) keeps modules that only write JSON/CSV
        free of the Rich dependency graph until this path is exercised—useful
        for minimal installs or tests that never call ``print_table``.

        Args:
            ranked: Ordered ``CandidateScore`` list (same order as written to JSON).

        Returns:
            None

        Raises:
            ImportError: If Rich is not installed when this method is called.

        Side effects:
            Writes formatted output to stdout via Rich's Console; no files.
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
        """Export ranked candidates to a wide CSV suitable for Excel review.

        Column set is the union of all requirement labels seen across
        candidates: each requirement becomes two columns (score + evidence).
        Candidates missing a given requirement get empty cells so every row
        aligns with the same header—critical for spreadsheet filters and pivots.

        Args:
            ranked: Ordered ``CandidateScore`` list.
            output_path: Destination ``.csv`` path.

        Returns:
            None

        Raises:
            OSError: If the file cannot be written.

        Side effects:
            Creates parent directories; overwrites the file; logs INFO.

        Notes:
            ``utf-8-sig`` writes a BOM so Excel on Windows recognises UTF-8;
            without it, non-ASCII text often displays incorrectly.
        """
        # dict.fromkeys preserves first-seen order while de-duplicating requirement names.
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
                # Index by requirement for O(1) lookup when filling wide rows.
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
                        # Same column count every row even when this CV had no row for that requirement.
                        row.append("")
                        row.append("")
                writer.writerow(row)

        logger.info("CSV written to %s", output_path)

    def _build_candidate_entry(self, rank: int, result: CandidateScore) -> dict:
        """Assemble one JSON object for the ``candidates`` array in ``write``.

        Keeps serialization logic in one place so JSON field names stay
        consistent and nested ``requirement_scores`` stay plain dicts (not model
        instances) for ``json.dump``.

        Args:
            rank: 1-based position in the sorted list.
            result: Scored candidate to flatten.

        Returns:
            A JSON-serialisable dict matching the project's output schema.

        Raises:
            None

        Side effects:
            None
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
