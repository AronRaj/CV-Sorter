"""CLI entry point for the CV Sorter multi-agent pipeline.

Primary workflow — no arguments required:
    python main.py

All configuration comes from .env, with sensible defaults for paths.
Optional flags exist for power users (--verbose, --export-csv, etc.)
but the default invocation just works.

Model assignments are fixed architectural decisions inside Supervisor:
  - Shortlist: Ollama llama3.1:8b (local)
  - JD extraction: Ollama gemma2:9b (local)
  - Scoring & Report: Claude Sonnet (API)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from src.agents.model_factory import CLAUDE_MODEL
from src.agents.supervisor import Supervisor
from src.core.config import OLLAMA_JD_MODEL, OLLAMA_SHORTLIST_MODEL, load_config
from src.core.models import CandidateScore, RequirementScore, Resume
from src.core.output_engine import OutputWriter


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    All arguments are optional with sensible defaults so the primary
    workflow is simply: python main.py

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="CV Sorter — rank candidates using a multi-agent LLM pipeline",
    )
    parser.add_argument(
        "--jd",
        default="job_description.txt",
        help="Path to job description text file (default: job_description.txt)",
    )
    parser.add_argument(
        "--resumes",
        default="src/resumes/",
        help="Folder containing PDF resumes (default: src/resumes/)",
    )
    parser.add_argument(
        "--output",
        default="src/results/ranked_output.json",
        help="Path for ranked JSON output (default: src/results/ranked_output.json)",
    )
    parser.add_argument(
        "--summary",
        default="src/results/recruiter_summary.md",
        help="Path for recruiter summary markdown (default: src/results/recruiter_summary.md)",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Also export results as CSV to this path",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=0,
        metavar="N",
        help="Only show candidates scoring >= N in terminal output",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print ranked table to terminal after run",
    )
    return parser


def _load_ranked_from_json(json_path: str) -> list[CandidateScore]:
    """Reconstruct CandidateScore objects from the ranked JSON output.

    Args:
        json_path: Path to the ranked JSON file written by Supervisor.

    Returns:
        List of CandidateScore instances in ranked order.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    results: list[CandidateScore] = []
    for entry in data.get("candidates", []):
        resume = Resume(
            filename=entry["filename"],
            raw_text="",
            name=None,
            skills=[],
            experience_years=None,
            education=[],
        )
        req_scores = [
            RequirementScore(
                requirement=rs["requirement"],
                score=rs["score"],
                evidence=rs["evidence"],
            )
            for rs in entry.get("requirement_scores", [])
        ]
        results.append(
            CandidateScore(
                resume=resume,
                overall_score=entry["overall_score"],
                fit_label=entry["fit_label"],
                explanation=entry["explanation"],
                requirement_scores=req_scores,
            )
        )
    return results


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


def _enable_langsmith_tracing(config: object) -> None:
    """Set LangSmith environment variables if tracing is enabled.

    Args:
        config: A Config instance with langchain_* fields.
    """
    if config.langchain_tracing_v2 == "true":
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = config.langchain_api_key or ""
        os.environ["LANGCHAIN_PROJECT"] = config.langchain_project


def main() -> None:
    """Parse arguments, validate inputs, and run the agent pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = load_config(verbose=args.verbose)
    except ValueError:
        print("Setup error: ANTHROPIC_API_KEY not found in .env")
        print("Add your Anthropic API key to .env and try again.")
        print("See README.md for setup instructions.")
        sys.exit(1)

    jd_path = Path(args.jd)
    if not jd_path.exists():
        print(f"Error: job description file not found: {args.jd}")
        print("Create job_description.txt in the project root.")
        sys.exit(1)

    resumes_folder = Path(args.resumes)
    if not resumes_folder.exists():
        print(f"Error: resumes folder not found: {args.resumes}")
        print("Create a resumes/ folder and add PDF files to it.")
        sys.exit(1)

    print()
    print("CV Sorter — Multi-Agent Recruitment Pipeline")
    print(f"  Shortlisting : Ollama {OLLAMA_SHORTLIST_MODEL} (local)")
    print(f"  JD analysis  : Ollama {OLLAMA_JD_MODEL} (local)")
    print(f"  Scoring      : Claude {CLAUDE_MODEL} (API)")
    print(f"  Report       : Claude {CLAUDE_MODEL} (API)")
    print()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    _enable_langsmith_tracing(config)

    try:
        supervisor = Supervisor(config=config)
        result = supervisor.run(
            jd_path=args.jd,
            resumes_folder=args.resumes,
            output_json_path=args.output,
            output_summary_path=args.summary,
        )
    except (ValueError, RuntimeError) as e:
        print(f"\nError: {e}")
        sys.exit(1)

    ranked = _load_ranked_from_json(result["ranked_output_path"])
    filtered = ranked

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

    print()
    print(f"  Results  → {result['ranked_output_path']}")
    print(f"  Summary  → {result['summary_path']}")
    print(f"  Elapsed  → {result['elapsed_seconds']}s")
    print()

    sys.exit(0)


if __name__ == "__main__":
    main()
