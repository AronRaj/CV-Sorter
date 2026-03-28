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

from model_factory import CLAUDE_MODEL
from supervisor import Supervisor
from config import OLLAMA_JD_MODEL, OLLAMA_SHORTLIST_MODEL, load_config
from models import CandidateScore, RequirementScore, Resume
from output_engine import OutputWriter


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface for CV Sorter.

    Defaults are chosen so evaluators and recruiters can run ``python main.py``
    without memorizing paths: JD and resume locations match the conventional
    layout documented in the project (root ``job_description.txt``, ``resumes/``).

    Returns:
        A fully configured ``argparse.ArgumentParser`` ready for ``parse_args()``.

    Side effects:
        None. This function only builds the parser; it does not read files or
        mutate global state.
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
        default="resumes/",
        help="Folder containing PDF resumes (default: resumes/)",
    )
    parser.add_argument(
        "--output",
        default="results/ranked_output.json",
        help="Path for ranked JSON output (default: results/ranked_output.json)",
    )
    parser.add_argument(
        "--summary",
        default="results/recruiter_summary.md",
        help="Path for recruiter summary markdown (default: results/recruiter_summary.md)",
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
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        metavar="KEY",
        help="Anthropic API key (overrides ANTHROPIC_API_KEY from .env / environment)",
    )
    return parser


def _load_ranked_from_json(json_path: str) -> list[CandidateScore]:
    """Hydrate in-memory score objects from the pipeline's ranked JSON artifact.

    The Supervisor already wrote structured results to disk; we reload them so
    optional post-steps (CSV export, terminal table) can reuse ``OutputWriter``
    without threading large objects through the return value of ``supervisor.run``.
    That keeps the supervisor focused on orchestration while the CLI owns
    presentation concerns.

    Args:
        json_path: Filesystem path to the ranked JSON file produced by the pipeline
            (same path passed as ``output_json_path`` to ``Supervisor.run``).

    Returns:
        Candidates in ranked order, each as a ``CandidateScore`` with nested
        ``RequirementScore`` rows where present in the JSON.

    Side effects:
        Reads the file at ``json_path``. Does not modify the file.

    Note:
        Reconstructed ``Resume`` instances use empty ``raw_text`` and minimal
        fields because downstream consumers here only need filename, scores, and
        explanations—not a second PDF parse. Full text lives in the JSON strings
        inside explanations if needed for debugging.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    results: list[CandidateScore] = []
    for entry in data.get("candidates", []):
        # Stub resume: satisfies dataclass/model shape for CSV/table writers
        # without duplicating heavy text extraction from PDFs.
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
    """Restrict terminal/CSV output to candidates meeting a score floor.

    The underlying JSON and markdown summary from the Supervisor are unchanged;
    this filter only affects what the CLI prints or exports when ``--min-score``
    is set. That lets recruiters focus a noisy run without re-running the LLMs.

    Args:
        ranked: Ordered list of ``CandidateScore`` (typically full ranked list).
        min_score: Inclusive minimum ``overall_score``; candidates below this are
            dropped from the returned list.

    Returns:
        A new list containing only candidates with ``overall_score >= min_score``.
        May be empty if every candidate falls below the threshold.

    Side effects:
        Prints a single warning line to stdout if filtering removes every
        candidate but the input list was non-empty, so the user knows the
        threshold was too aggressive rather than the pipeline failing.
    """
    filtered = [c for c in ranked if c.overall_score >= min_score]

    if not filtered and ranked:
        print(f"Warning: --min-score {min_score} removed all candidates.")

    return filtered


def _enable_langsmith_tracing(config: object) -> None:
    """Activate LangChain/LangSmith tracing for the current process when configured.

    Tracing is opt-in via ``.env`` so local runs stay private by default; when
    enabled, child processes and LangChain calls inherit these environment
    variables automatically.

    Args:
        config: Loaded application config (expected to expose ``langchain_tracing_v2``,
            ``langchain_api_key``, and ``langchain_project`` as loaded by
            ``load_config``).

    Returns:
        None.

    Side effects:
        Mutates ``os.environ`` for the current Python process when tracing is on.
        No-op when ``langchain_tracing_v2`` is not the string ``"true"``.
    """
    if config.langchain_tracing_v2 == "true":
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = config.langchain_api_key or ""
        os.environ["LANGCHAIN_PROJECT"] = config.langchain_project


def main() -> None:
    """Run the end-to-end CLI: validate inputs, execute agents, then optional exports.

    Flow:
        1. Parse CLI flags and load ``.env``-backed config (hard requirement on
           Anthropic API key because scoring/report stages use Claude).
        2. Fail fast if JD file or resumes folder is missing—cheaper than
           starting local Ollama work only to abort.
        3. Print a short banner so logs clearly show which models back each
           stage (local vs API), which matters for reproducibility and grading.
        4. Run ``Supervisor``; on failure, surface the error and exit non-zero.
        5. Reload ranked JSON for optional filtering, CSV export, and verbose
           table output; always print output paths and elapsed time.

    Returns:
        None. The process exits via ``sys.exit`` with code ``0`` on success,
        ``1`` on setup, validation, or pipeline errors.

    Side effects:
        Writes pipeline outputs (JSON, markdown) via ``Supervisor``; may write
        CSV; prints to stdout/stderr; configures the root logger; may set
        LangSmith-related environment variables.
    """
    # --- CLI definition and environment-backed configuration ---
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(verbose=args.verbose)

    # CLI --api-key overrides .env / environment so evaluators can pass the key
    # as a command-line parameter without editing any config files.
    if args.api_key:
        config.anthropic_api_key = args.api_key

    if not config.anthropic_api_key:
        print("Setup error: ANTHROPIC_API_KEY not provided.")
        print("Supply it via one of:")
        print("  1. --api-key flag:  python main.py --api-key <KEY>")
        print("  2. Environment var: export ANTHROPIC_API_KEY=<KEY>")
        print("  3. .env file:       ANTHROPIC_API_KEY=<KEY>")
        sys.exit(1)

    # --- Required inputs: fail before any model or file I/O from the pipeline ---
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

    # --- Model disclosure (matches architectural split inside Supervisor) ---
    print()
    print("CV Sorter — Multi-Agent Recruitment Pipeline")
    print(f"  Shortlisting : Ollama {OLLAMA_SHORTLIST_MODEL} (local)")
    print(f"  JD analysis  : Ollama {OLLAMA_JD_MODEL} (local)")
    print(f"  Scoring      : Claude {CLAUDE_MODEL} (API)")
    print(f"  Report       : Claude {CLAUDE_MODEL} (API)")
    print()

    # Minimal format: line-oriented logs read cleanly next to the banner above.
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    _enable_langsmith_tracing(config)

    # --- Core pipeline (all heavy work: parse JD, shortlist, score, write outputs) ---
    try:
        supervisor = Supervisor(config=config)
        result = supervisor.run(
            jd_path=args.jd,
            resumes_folder=args.resumes,
            output_json_path=args.output,
            output_summary_path=args.summary,
        )
    except (ValueError, RuntimeError) as e:
        # ValueError: bad inputs or config; RuntimeError: wrapped failures from agents/tools.
        print(f"\nError: {e}")
        sys.exit(1)

    # Reload from disk so CSV/table paths share one code path with the saved artifact.
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
