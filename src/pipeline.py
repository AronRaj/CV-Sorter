"""Pipeline orchestrator for the CV Sorter.

Calls all modules in the correct order: JD parsing → resume parsing →
scoring → sorting → output. Contains zero business logic — only coordination.
"""

from __future__ import annotations

import logging

from src.config import Config
from src.jd_parser import JDParser
from src.llm_client import LLMClient
from src.models import CandidateScore
from src.output import OutputWriter
from src.parser import PDFParser, ResumeParser
from src.scorer import Scorer

logger = logging.getLogger(__name__)

MODEL_MAP: dict[str, str] = {
    "claude": "claude-sonnet-4-5",
    "openai": "gpt-4o",
    "gemini": "gemini-2.0-flash",
}


class Pipeline:
    """Orchestrates the full CV scoring pipeline.

    Coordinates all modules in sequence: parse JD, parse resumes,
    score each candidate, sort by score, and write output. No business
    logic lives here — each step delegates to the responsible module.
    """

    def __init__(self, config: Config) -> None:
        """Initialise with the application configuration.

        Args:
            config: The loaded Config instance.
        """
        self._config = config

    def run(
        self,
        jd_path: str,
        resumes_folder: str,
        provider: str,
        output_path: str,
        verbose: bool = False,
    ) -> list[CandidateScore]:
        """Execute the full scoring pipeline.

        Steps executed in order:
        1. Build LLM client for the chosen provider
        2. Parse the job description via LLM
        3. Parse all resume PDFs from the folder
        4. Score each resume against the JD (skips failures)
        5. Sort by overall_score descending
        6. Write JSON output and optionally print Rich table

        Args:
            jd_path: Path to the job description text file.
            resumes_folder: Path to the folder containing resume PDFs.
            provider: LLM provider name (e.g. "groq", "claude").
            output_path: File path for the output JSON.
            verbose: If True, print a Rich ranking table to stdout.

        Returns:
            The ranked list of CandidateScore instances.

        Raises:
            RuntimeError: If no resumes are found or all scoring fails.
        """
        model = MODEL_MAP.get(provider, self._config.groq_model)
        logger.info("CV Sorter starting — provider: %s | model: %s", provider, model)

        client = LLMClient.build(provider=provider, config=self._config)

        logger.info("Parsing job description...")
        jd_parser = JDParser(client=client)
        jd = jd_parser.parse(jd_path)
        logger.info("  Job title: %s", jd.job_title)

        logger.info("Parsing resumes...")
        pdf_parser = PDFParser.build(self._config.pdf_parser)
        resume_parser = ResumeParser(parser=pdf_parser)
        resumes = resume_parser.parse_all(resumes_folder)
        logger.info("  %d resume(s) found", len(resumes))

        if not resumes:
            raise RuntimeError(
                f"No PDF files found in '{resumes_folder}'. "
                "Add at least one PDF resume and try again."
            )

        logger.info("Scoring candidates...")
        scorer = Scorer(client=client)
        scored: list[CandidateScore] = []

        for resume in resumes:
            try:
                logger.info("  Scoring %s...", resume.filename)
                result = scorer.score(resume=resume, jd=jd)
                scored.append(result)
            except RuntimeError as e:
                logger.warning(
                    "  Skipping %s — scoring failed: %s",
                    resume.filename,
                    e,
                )

        if not scored:
            raise RuntimeError(
                "All candidates failed scoring. Check logs for details."
            )

        ranked = sorted(scored, key=lambda c: c.overall_score, reverse=True)

        logger.info("Writing results...")
        writer = OutputWriter()
        writer.write(
            ranked=ranked,
            jd=jd,
            provider=provider,
            model=model,
            output_path=output_path,
        )

        if verbose:
            writer.print_table(ranked)

        return ranked
