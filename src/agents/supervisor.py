"""Supervisor — sequential coordinator for the multi-agent pipeline.

Calls Shortlist -> Scorer -> Report in fixed order, passes typed data
between agents explicitly, and logs timing for each step.  This is
plain Python coordination, not a LangGraph graph or AgentExecutor.

Fixed model assignments (architectural decisions):
  - Shortlist: Ollama llama3.1:8b (local, zero API cost)
  - JD extraction: Ollama gemma2:9b (local, zero API cost)
  - Scoring: Claude Sonnet (Anthropic API)
  - Report: Claude Sonnet (Anthropic API)
"""

from __future__ import annotations

import logging
import time

from src.agents.model_factory import (
    get_claude_model,
    get_ollama_jd_model,
    get_ollama_shortlist_model,
)
from src.agents.report_agent import ReportAgent
from src.agents.scorer_agent import ScorerAgent
from src.agents.shortlist_agent import ShortlistAgent
from src.core.config import Config
from src.core.jd_parser_engine import JDParser
from src.core.output_engine import OutputWriter
from src.core.parser_engine import PDFParser, ResumeParser

logger = logging.getLogger(__name__)


class Supervisor:
    """Sequential coordinator for the multi-agent pipeline.

    Builds all models internally using the named factory functions.
    The caller passes only a Config — model assignments are fixed
    architectural decisions, not runtime options.
    """

    def __init__(self, config: Config) -> None:
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
        output_json_path: str = "src/results/agent_ranked_output.json",
        output_summary_path: str = "src/results/recruiter_summary.md",
    ) -> dict:
        """Run the full multi-agent pipeline.

        Args:
            jd_path:             Path to the job description .txt file.
            resumes_folder:      Path to the folder containing PDF resumes.
            output_json_path:    Where to write the ranked JSON output.
            output_summary_path: Where to write the recruiter summary.

        Returns:
            A dict with pipeline results and metadata.

        Raises:
            RuntimeError: If no PDF files are found in resumes_folder.
        """
        start = time.time()
        logger.info("[Supervisor]  Starting agent pipeline...")

        # --- Step 1: Parse inputs using existing src/ modules ---
        logger.info("[Supervisor]  Parsing inputs...")
        pdf_parser = PDFParser.build(self._config.pdf_parser)
        resumes = ResumeParser(parser=pdf_parser).parse_all(resumes_folder)
        if not resumes:
            raise RuntimeError(
                f"No PDF files found in '{resumes_folder}'. "
                "Add at least one PDF resume."
            )

        jd = JDParser(model=self._jd_model).parse(jd_path)
        logger.info(
            "[Supervisor]  Parsed %d resume(s), JD: %s",
            len(resumes),
            jd.job_title,
        )

        # --- Step 2: Shortlist ---
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

        if not shortlisted:
            logger.warning(
                "[Supervisor]  All candidates skipped — "
                "falling back to scoring all resumes."
            )
            shortlisted = resumes

        # --- Step 3: Score shortlisted candidates ---
        logger.info("[Supervisor]  Running scorer agent...")
        scored = self._scorer_agent.run(
            resumes=shortlisted,
            jd=jd,
        )

        # --- Step 4: Generate report ---
        logger.info("[Supervisor]  Running report agent...")
        summary_path = self._report_agent.run(
            results=scored,
            jd=jd,
            output_path=output_summary_path,
        )

        # --- Step 5: Write ranked JSON using existing OutputWriter ---
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
