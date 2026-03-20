"""CLI entry point for the CV Sorter.

Parses command-line arguments, validates inputs, and runs the
scoring pipeline. No business logic lives here.

Usage:
    python main.py --jd job_description.txt --resumes ./resumes --provider groq --verbose

Note: --min-score and --filter-skills filter the terminal display
and CSV export only. The JSON output always contains all candidates.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import load_config
from src.models import CandidateScore
from src.output import OutputWriter
from src.pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description=(
            "CV Sorter: rank candidate resumes against a job description using LLMs. "
            "Note: --min-score and --filter-skills filter the terminal display "
            "and CSV export only. The JSON output always contains all candidates."
        ),
    )
    parser.add_argument(
        "--jd",
        required=True,
        help="Path to job description .txt file",
    )
    parser.add_argument(
        "--resumes",
        required=True,
        help="Path to folder containing PDF resumes",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["claude", "openai", "gemini", "groq"],
        help="LLM provider to use (default: from .env DEFAULT_PROVIDER)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for output JSON file (default: from .env OUTPUT_PATH)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a rich results table to the terminal",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Also write a CSV file to this path (e.g. results/ranked.csv)",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=0,
        metavar="N",
        help="Only show candidates with overall score >= N (default: 0, show all)",
    )
    parser.add_argument(
        "--filter-skills",
        type=str,
        default=None,
        metavar="SKILLS",
        help=(
            'Comma-separated skills to filter by. Only show candidates who '
            'scored >= 50 on ALL listed skills. Example: "Python,Docker"'
        ),
    )
    return parser


def _apply_skill_filter(
    ranked: list[CandidateScore],
    skills_csv: str,
) -> list[CandidateScore]:
    """Keep only candidates scoring >= 50 on ALL listed skills.

    Skill matching is case-insensitive and uses substring containment,
    so "python" matches a requirement named "Python 5+ years".

    Args:
        ranked: Full ranked list of candidates.
        skills_csv: Comma-separated skill names from --filter-skills.

    Returns:
        Filtered list (may be empty).
    """
    required_skills = [s.strip().lower() for s in skills_csv.split(",") if s.strip()]
    if not required_skills:
        return ranked

    filtered: list[CandidateScore] = []
    for candidate in ranked:
        passes_all = True
        for skill in required_skills:
            matched = any(
                skill in rs.requirement.lower() and rs.score >= 50
                for rs in candidate.requirement_scores
            )
            if not matched:
                passes_all = False
                break
        if passes_all:
            filtered.append(candidate)

    if not filtered:
        print(
            "Warning: --filter-skills removed all candidates. "
            "Try broader skill names or a lower threshold."
        )

    return filtered


def _apply_min_score(
    ranked: list[CandidateScore],
    min_score: int,
) -> list[CandidateScore]:
    """Keep only candidates with overall_score >= min_score.

    Args:
        ranked: Ranked list of candidates (possibly already filtered).
        min_score: Minimum overall score threshold.

    Returns:
        Filtered list (may be empty).
    """
    filtered = [c for c in ranked if c.overall_score >= min_score]

    if not filtered and ranked:
        print(f"Warning: --min-score {min_score} removed all candidates.")

    return filtered


def main() -> None:
    """Parse arguments, validate inputs, and run the pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = load_config(verbose=args.verbose)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    provider = args.provider if args.provider else config.default_provider
    output_path = args.output if args.output else config.output_path

    jd_path = Path(args.jd)
    if not jd_path.exists():
        print(f"Error: job description file not found: {args.jd}")
        sys.exit(1)

    resumes_folder = Path(args.resumes)
    if not resumes_folder.exists():
        print(f"Error: resumes folder not found: {args.resumes}")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    try:
        pipeline = Pipeline(config=config)
        ranked = pipeline.run(
            jd_path=args.jd,
            resumes_folder=args.resumes,
            provider=provider,
            output_path=output_path,
            verbose=False,
        )
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    filtered = ranked

    if args.filter_skills:
        filtered = _apply_skill_filter(
            ranked=filtered,
            skills_csv=args.filter_skills,
        )

    if args.min_score > 0:
        filtered = _apply_min_score(
            ranked=filtered,
            min_score=args.min_score,
        )

    if args.export_csv:
        OutputWriter().write_csv(
            ranked=filtered,
            output_path=args.export_csv,
        )

    if args.verbose:
        OutputWriter().print_table(filtered)

    sys.exit(0)


if __name__ == "__main__":
    main()
