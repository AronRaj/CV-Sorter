"""Supervisor — sequential coordinator for the multi-agent pipeline.

This module implements the orchestration layer: it is **plain Python
coordination**, not a LangGraph graph or LangChain AgentExecutor. Steps run
in a fixed order with explicit hand-offs of typed domain objects
(`Resume`, `JobDescription`, `CandidateScore`), which keeps the flow easy to
audit and avoids implicit tool-routing bugs.

**Multi-agent architecture and model choices**

- **Shortlist (Ollama llama3.1:8b):** Runs once per resume for a cheap binary
  PROCEED/SKIP gate. Local inference avoids API cost for the largest batch.
- **JD extraction (Ollama gemma2:9b):** Structured field extraction from the JD
  file; separated from shortlist so each task can use a model tuned for that
  style of output without paying cloud fees.
- **Scorer (Claude Sonnet):** Deep, evidence-linked scoring over full resume
  text; needs strong reasoning and long context, so it uses the paid API only
  on the (smaller) shortlisted set.
- **Report (Claude Sonnet):** Synthesizes *all* scored candidates into one
  briefing; same high-quality model as scoring because cross-candidate prose
  and comparisons are the product recruiters read.

**Data flow (high level)**

1. PDFs + JD path → parsers → `list[Resume]` + `JobDescription`.
2. Shortlist agent → `(shortlisted, skipped)`; empty shortlist triggers a
   deliberate fallback (score everyone) so the pipeline never dead-ends.
3. Scorer agent → `list[CandidateScore]` sorted by overall score.
4. Report agent → writes markdown summary; `OutputWriter` writes ranked JSON.

Timing is logged per major step to support evaluation and performance tuning.
"""

from __future__ import annotations

import logging
import time

from model_factory import (
    get_claude_model,
    get_ollama_jd_model,
    get_ollama_shortlist_model,
)
from report_agent import ReportAgent
from scorer_agent import ScorerAgent
from shortlist_agent import ShortlistAgent
from config import Config
from jd_parser_engine import JDParser
from output_engine import OutputWriter
from parser_engine import PDFParser, ResumeParser

logger = logging.getLogger(__name__)


class Supervisor:
    """Sequential coordinator for the multi-agent CV ranking pipeline.

    Instantiates chat models via the factory (fixed assignments — not
    user-selectable at runtime) and wires **ShortlistAgent → ScorerAgent →
    ReportAgent**. The supervisor does not call the LLM itself except indirectly
    through `JDParser`, which uses the injected JD model.

    **Side effects:** `run()` triggers file writes (report markdown, ranked JSON)
    and depends on disk paths in `Config` / arguments for parsers and outputs.
    """

    def __init__(self, config: Config) -> None:
        """Build models and agents from application configuration.

        Claude is shared by scorer and report so both stages see consistent
        reasoning style; Ollama models are separate instances for shortlist vs
        JD parsing.

        Args:
            config: Loaded `Config` (must include `anthropic_api_key` for API
                agents; Ollama is expected reachable at default host/port).

        Side effects:
            Reads prompt templates inside agent constructors; may validate API
            key when models are first used (factory raises if key missing).
        """
        self._config = config

        claude_model = get_claude_model(config)
        shortlist_model = get_ollama_shortlist_model()
        jd_model = get_ollama_jd_model()

        self._shortlist_agent = ShortlistAgent(model=shortlist_model)
        self._scorer_agent = ScorerAgent(model=claude_model)
        self._report_agent = ReportAgent(model=claude_model)
        self._jd_model = jd_model

    def run(
        self,
        jd_path: str,
        resumes_folder: str,
        output_json_path: str = "results/ranked_output.json",
        output_summary_path: str = "results/recruiter_summary.md",
    ) -> dict:
        """Execute the full pipeline from raw inputs to ranked output and summary.

        Order is strict: parse → shortlist → score → report → persist JSON.
        Each stage receives outputs from the previous one; there is no parallel
        fan-out, which simplifies debugging and matches capstone evaluation
        expectations.

        Args:
            jd_path: Path to the job description text file consumed by
                `JDParser`.
            resumes_folder: Directory scanned for PDF resumes (via
                `ResumeParser`).
            output_json_path: Destination for machine-readable ranked results
                (`OutputWriter`).
            output_summary_path: Destination for the recruiter markdown briefing
                (`ReportAgent`).

        Returns:
            A dict with:
            - ``ranked_output_path``: Path written for JSON ranked output.
            - ``summary_path``: Path written for markdown summary (same as
              ``output_summary_path``).
            - ``total_candidates``: Count of PDFs parsed from the folder.
            - ``shortlisted_count``: Resumes passed to the scorer after
              shortlisting (may equal total after empty-shortlist fallback).
            - ``skipped_count``: Count from shortlist agent (still reported even
              if fallback rescored everyone).
            - ``elapsed_seconds``: Wall-clock seconds for the full run.

        Raises:
            RuntimeError: If no PDF files exist under ``resumes_folder``.

        Side effects:
            Creates output directories as needed, writes JSON and markdown files,
            and emits structured log lines for each phase.
        """
        start = time.time()
        logger.info("[Supervisor]  Starting agent pipeline...")

        # ------------------------------------------------------------------
        # Step 1: Deterministic ingestion — no LLM in PDF parsing; JD uses LLM.
        # ------------------------------------------------------------------
        logger.info("[Supervisor]  Parsing inputs...")
        pdf_parser = PDFParser.build(self._config.pdf_parser)
        resumes = ResumeParser(parser=pdf_parser).parse_all(resumes_folder)
        if not resumes:
            raise RuntimeError(
                f"No PDF files found in '{resumes_folder}'. "
                "Add at least one PDF resume."
            )

        # JDParser shares the same architectural pattern as agents: model
        # injected for testability and to keep extraction logic in one place.
        jd = JDParser(model=self._jd_model).parse(jd_path)
        logger.info(
            "[Supervisor]  Parsed %d resume(s), JD: %s",
            len(resumes),
            jd.job_title,
        )

        # ------------------------------------------------------------------
        # Step 2: Shortlist — reduce API spend before Claude scoring.
        # ------------------------------------------------------------------
        logger.info("[Supervisor]  Running shortlist agent...")
        shortlisted, skipped = self._shortlist_agent.run(
            resumes=resumes,
            jd=jd,
        )
        logger.info(
            "[Supervisor]  Shortlisted: %d, Skipped: %d",
            len(shortlisted),
            len(skipped),
        )

        # If every resume was skipped (or errors collapsed the list), scoring
        # nobody would make the product useless; fall back to the full set so
        # recruiters always get an ordering unless parsing failed earlier.
        if not shortlisted:
            logger.warning(
                "[Supervisor]  All candidates skipped — "
                "falling back to scoring all resumes."
            )
            shortlisted = resumes

        # ------------------------------------------------------------------
        # Step 3: Deep score — Claude, structured JSON + self-eval inside agent.
        # ------------------------------------------------------------------
        logger.info("[Supervisor]  Running scorer agent...")
        scored = self._scorer_agent.run(
            resumes=shortlisted,
            jd=jd,
        )

        # ------------------------------------------------------------------
        # Step 4: Cross-candidate report — needs full scored list in one prompt.
        # ------------------------------------------------------------------
        logger.info("[Supervisor]  Running report agent...")
        summary_path = self._report_agent.run(
            results=scored,
            jd=jd,
            output_path=output_summary_path,
        )

        # ------------------------------------------------------------------
        # Step 5: JSON artifact — separate from markdown for downstream tools.
        # ------------------------------------------------------------------
        OutputWriter().write(
            ranked=scored,
            jd=jd,
            provider="multi-agent",
            model="shortlist:ollama + score/report:api",
            output_path=output_json_path,
        )

        elapsed = round(time.time() - start, 1)
        logger.info(
            "[Supervisor]  Done in %ss. Results → %s",
            elapsed,
            output_json_path,
        )

        return {
            "ranked_output_path": output_json_path,
            "summary_path": summary_path,
            "total_candidates": len(resumes),
            "shortlisted_count": len(shortlisted),
            "skipped_count": len(skipped),
            "elapsed_seconds": elapsed,
        }
