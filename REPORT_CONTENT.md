# CV Sorting Using LLMs — Capstone Report

---

## Abstract

Manual resume screening is one of the most time-consuming tasks in recruitment. Hiring managers spend an average of six to eight seconds per resume, leading to inconsistent evaluations and qualified candidates being overlooked. This project presents CV Sorter, a Python-based tool that automates resume ranking by leveraging Large Language Models (LLMs). The tool ingests a job description and a folder of candidate resumes in PDF format, then uses an LLM to score each resume on a 0–100 scale across every requirement listed in the job description. Each score is accompanied by a textual evidence citation drawn directly from the resume, providing transparency and explainability. The system supports multiple LLM providers — Groq (Llama 3.3 70B), OpenAI GPT-4o, and Google Gemini — through a pluggable abstract interface, enabling cross-provider comparison of scoring behaviour. Beyond the command-line interface, the project includes a Streamlit-based recruiter dashboard that displays ranked candidates with expandable evidence drill-down and interactive re-ranking via per-requirement weight sliders — all without additional LLM calls. CSV export and CLI-level filtering flags (--min-score, --filter-skills) provide recruiter-friendly data export and shortlisting capabilities. Testing with two candidate resumes against a Senior Backend Engineer job description showed consistent ranking across providers, with overall scores varying by only 3–5 points between Groq and OpenAI. The project demonstrates that LLMs can produce structured, evidence-based resume evaluations suitable for augmenting — not replacing — human recruitment decisions.

---

## 1. Introduction

The hiring process in modern organisations begins with resume screening — a largely manual activity where recruiters evaluate dozens or hundreds of candidate resumes against a job description. This process is inherently subjective; different reviewers may weigh the same qualification differently, and fatigue from reading many similar documents introduces inconsistency. As organisations scale, the volume of applications makes thorough manual screening impractical.

Recent advances in Large Language Models (LLMs) have made it possible to automate structured text analysis tasks that previously required human judgement. LLMs can read a resume, identify relevant skills and experience, compare them against job requirements, and produce a scored evaluation — all within seconds and at a fraction of the cost of human review time.

This project, CV Sorter, explores the application of LLMs to resume ranking. It is a Python-based tool that takes a job description and a set of candidate resumes as input, sends each resume to an LLM for evaluation, and produces a ranked list of candidates with per-requirement scores and textual evidence. The tool is designed around two key principles: transparency (every score must cite specific evidence from the resume) and pluggability (the LLM provider and PDF parser can be swapped without changing business logic). In addition to the CLI, the project provides a Streamlit web dashboard that enables recruiters to explore results interactively, filter candidates by score or skill, and re-rank candidates by adjusting requirement weights — all from cached scoring data with zero additional LLM calls.

The project was built incrementally over four days: Day 1 established the data models and parsing layer, Day 2 built the scoring engine and CLI pipeline, Day 3 added a test suite and edge-case hardening, and Day 4 added the presentation and interactivity layer including CSV export, CLI filter flags, and the Streamlit dashboard. This report documents the design decisions, implementation, results, and lessons learned.

---

## 2. Problem Statement

Recruiting teams face a critical bottleneck at the resume screening stage. For a typical software engineering role, a company may receive 100–500 applications. A recruiter spending two minutes per resume would need over 16 hours of focused reading to screen 500 candidates — and this is before any interviews take place.

The core problems with manual screening are:

**Inconsistency.** Two recruiters evaluating the same resume against the same job description will often reach different conclusions. Fatigue, implicit bias, and subjective interpretation of qualifications all contribute to inconsistent outcomes.

**Lack of evidence.** When a recruiter shortlists or rejects a candidate, the reasoning is rarely documented in a structured way. This makes it difficult to audit decisions or provide constructive feedback to candidates.

**Scalability.** Manual screening does not scale with application volume. As companies grow and roles attract more applicants, the quality of screening degrades unless proportionally more recruiter time is allocated.

**Inflexible prioritisation.** Different hiring managers may prioritise different skills for the same role. A backend team lead may value PostgreSQL experience more heavily than Docker, while a DevOps-oriented manager may do the opposite. Static scoring systems cannot accommodate these preferences without re-running the entire evaluation.

This project addresses these problems by delegating the structured comparison of resumes against job requirements to an LLM, and providing an interactive dashboard where recruiters can adjust skill weights and re-rank candidates without additional LLM calls. The system does not replace human decision-making; rather, it produces a ranked shortlist with explicit evidence for each score, enabling recruiters to focus their time on the most promising candidates.

---

## 3. Objectives

The objectives of this project are:

- **Build a CLI tool** that accepts a job description and a folder of PDF resumes, and outputs a ranked JSON file with per-requirement scores and evidence citations.

- **Implement a pluggable LLM architecture** supporting multiple providers (Groq/Llama, OpenAI GPT-4o, Google Gemini) through a single abstract interface, enabling provider comparison without code changes.

- **Extract structured data from PDFs** using pymupdf4llm to convert resume content into markdown that preserves formatting, tables, and multi-column layouts for optimal LLM comprehension.

- **Design an extensible PDF parsing layer** using the abstract factory pattern, with a pre-designed extension path for PaddleOCR to handle scanned/image-based PDFs as a two-file change.

- **Produce transparent, evidence-based evaluations** where every requirement score includes a textual evidence field citing specific content from the candidate's resume.

- **Implement automatic retry logic** for LLM JSON parsing failures, using a stricter prompt on retry to handle the inherent non-determinism of LLM output formatting.

- **Validate the system with a comprehensive test suite** (21 unit and integration tests) that runs without API calls using mock LLM clients and monkeypatched environment variables.

- **Compare scoring behaviour across LLM providers** to identify differences in evaluation stringency, evidence quality, and ranking consistency.

- **Provide a recruiter-facing user interface** with a Streamlit web dashboard featuring ranked candidate cards, expandable evidence drill-down, sidebar filters, and interactive re-ranking via per-requirement weight sliders.

- **Support multiple output formats and CLI filtering** including CSV export (--export-csv), minimum score threshold (--min-score), and skill-based filtering (--filter-skills) for recruiter-friendly data export and shortlisting.

---

## 4. Methodology

The system was implemented in Python 3.10+ following a modular architecture with strict separation of concerns. Each module has a defined input/output contract, and dependencies flow in one direction: the pipeline orchestrator calls individual modules, but individual modules never import from the pipeline.

**Data Models.** All data structures are defined as Python dataclasses in a single `models.py` file: `Resume`, `JobDescription`, `RequirementScore`, and `CandidateScore`. This mirrors the data class pattern from Kotlin/Java and serves as the single source of truth for all data shapes exchanged between modules.

**PDF Parsing.** Resume PDFs are parsed using pymupdf4llm, which outputs markdown rather than plain text. This preserves structural information — tables become markdown tables, headings are marked, and multi-column layouts are correctly ordered. The parser is accessed through an abstract `PDFParser` interface with a factory method, allowing PaddleOCR to be plugged in for scanned PDFs without changing any calling code.

**LLM Integration.** All LLM providers implement a single `complete(prompt: str) -> str` interface. The Groq provider uses the OpenAI-compatible API with a custom base URL, avoiding an additional SDK dependency. Prompt templates are stored in external `.txt` files and filled using Python string formatting, enabling prompt iteration without code changes.

**Scoring Pipeline.** The scorer fills a prompt template with the job title, required skills, responsibilities, and the candidate's resume text, then parses the LLM's JSON response into a `CandidateScore`. If JSON parsing fails, the system retries once with a stricter prompt that includes the failed response for context. The pipeline orchestrator coordinates the full flow: parse JD → parse resumes → score each → sort by score → write output.

**Output Formats.** The system produces three output formats: (1) a structured JSON file containing the full ranked list with all per-requirement scores and evidence, (2) a flat CSV file suitable for import into Excel or Google Sheets, and (3) a colour-coded Rich terminal table for quick command-line review. The JSON output always contains all candidates unfiltered; the CSV and terminal table respect any --min-score or --filter-skills flags provided at the command line.

**Streamlit Dashboard.** A Streamlit web application (`streamlit_app.py`) provides a recruiter-facing interface that reads the JSON output produced by the pipeline. The dashboard displays ranked candidate cards with expandable evidence drill-down per requirement, using colour-coded score badges (green for 80+, blue for 60–79, orange for 40–59, red for 0–39). Sidebar controls provide a minimum score slider and a skill filter text input for live filtering. An interactive re-ranking section allows recruiters to adjust per-requirement weight sliders (0.0× to 2.0×) and re-sort candidates using a weighted score formula — all computed client-side from cached JSON data with zero additional LLM calls. This separation ensures that the dashboard is instant, free to use, and does not depend on API availability.

**Re-ranking Algorithm.** The interactive re-ranking computes a weighted normalised score for each candidate: for each requirement, the raw score is multiplied by the slider weight, and the results are summed and divided by the total weight. Setting a weight to 2.0× doubles that requirement's contribution; setting it to 0.0× removes it entirely. Fit labels are re-derived from the new weighted score using the same bands as the scorer (80–100 Strong match, 60–79 Good match, 40–59 Partial match, 0–39 Weak match). This allows recruiters to prioritise their most important requirements without re-running the LLM.

**Testing.** The test suite uses mock LLM clients returning hardcoded JSON strings, pytest monkeypatch for environment isolation, and a synthetic PDF for parser tests. All 21 tests run in under one second with zero API calls. Streamlit UI testing was out of scope as it requires additional tooling (e.g. Selenium or Streamlit's AppTest) beyond the project's requirements.

---

## 5. User Interface

The system provides two interfaces: a command-line interface (CLI) for batch processing and a Streamlit web dashboard for interactive exploration.

### 5.1 Command-Line Interface

The CLI is the primary interface for running the scoring pipeline. It accepts a job description file and a resumes folder, calls the LLM to score each candidate, and produces ranked output in JSON format. Three additional flags extend its functionality:

- **--export-csv PATH** writes a flat CSV file alongside the JSON output, suitable for import into Excel or Google Sheets.

- **--min-score N** filters the terminal table and CSV export to show only candidates with overall scores at or above the specified threshold. The JSON file is unaffected.

- **--filter-skills "Python,Docker"** filters by skill, showing only candidates who scored 50 or above on all listed skills. Matching is case-insensitive and uses substring containment (e.g., "python" matches "Python 5+ years").

These flags are applied after the pipeline completes, ensuring the full unfiltered results are always preserved in the JSON output for later analysis or dashboard use.

### 5.2 Streamlit Dashboard

The Streamlit dashboard (`streamlit_app.py`) provides a web-based interface for recruiters to explore scoring results interactively. It loads a ranked JSON file produced by the CLI and displays three main sections:

**Candidate Cards.** Each candidate is shown in an expandable card displaying their rank, filename, overall score, and fit label. Expanding a card reveals the LLM's explanation and a per-requirement breakdown with colour-coded score badges and evidence citations.

**Sidebar Filters.** A minimum score slider (0–100) and a skill filter text input allow recruiters to narrow the displayed candidates in real time. Filters update live as controls are adjusted — no button click required.

**Interactive Re-ranking.** Below the candidate cards, a set of sliders (one per requirement, range 0.0× to 2.0×, default 1.0×) allows recruiters to adjust the importance of each requirement. Clicking the Re-rank button recomputes weighted scores and re-sorts candidates instantly. Rank changes are annotated (e.g., "was #2"). This feature enables different hiring managers to apply their own prioritisation without re-running the LLM.

---

## 6. Results and Analysis

The system was tested using two candidate resumes — Ayush Kushwaha (Android developer) and Saumitra (Android developer at Samsung) — against a Senior Backend Engineer job description requiring Python, REST API design, PostgreSQL, Docker, and cloud platform experience.

### Cross-Provider Scoring Comparison

| Provider          | Model                    | Ayush (Score) | Saumitra (Score) | Rank Order |
|-------------------|--------------------------|---------------|------------------|------------|
| Groq              | Llama 3.3 70B Versatile  | 20/100        | 12/100           | Ayush > Saumitra |
| OpenAI            | GPT-4o                   | 20/100        | 15/100           | Ayush > Saumitra |

### Key Observations

**Observation 1: Ranking order is consistent across providers.** Both Groq (Llama 3.3 70B) and OpenAI (GPT-4o) ranked Ayush Kushwaha above Saumitra. Both candidates were correctly classified as "Weak match" for a Senior Backend Engineer role, which is accurate given their Android development backgrounds. This consistency suggests that LLMs converge on relative ordering even when absolute scores differ.

**Observation 2: Absolute scores differ by a small margin (3–8 points).** For Ayush, both providers assigned an overall score of 20. For Saumitra, Groq scored 12 while OpenAI scored 15 — a 3-point difference. The variation comes from how each model weighs partial evidence: OpenAI awarded higher scores for REST API and cross-team collaboration, while Groq was stricter on requiring backend-specific context for those skills.

**Observation 3: Evidence quality varies between providers.** OpenAI's evidence text was more specific and contextual. For example, for the REST API requirement, OpenAI noted "Incorporated RESTful APIs in Android projects, but not primarily focused on backend REST API design" (score: 40), while Groq wrote "The candidate mentions 'RESTfulAPI' in their skills section" (score: 20). OpenAI's evidence better distinguishes between having used an API and having designed one, which is more useful for a recruiter reviewing the results.

### Per-Requirement Score Breakdown (Ayush Kushwaha)

| Requirement                    | Groq Score | OpenAI Score |
|--------------------------------|------------|--------------|
| Python                         | 20         | 30           |
| REST API design                | 30         | 40           |
| PostgreSQL                     | 0          | 0            |
| Docker                         | 0          | 0            |
| Cloud platforms                | 0          | 0            |
| Design scalable microservices  | 10         | 0            |
| Improve reliability/performance| 20         | 30           |
| Mentor junior engineers        | 10         | 0            |
| Collaborate with teams         | 20         | 20           |

Both providers correctly identified zero evidence for PostgreSQL, Docker, and cloud platforms — requirements completely absent from the Android-focused resumes. The highest-scored requirements (REST API, Python) correspond to skills the candidate demonstrated in side projects, not professional experience, which both LLMs appropriately noted in their evidence text.

### Interactive Re-ranking Demonstration

The Streamlit dashboard's re-ranking feature was tested by adjusting requirement weights. For example, setting the REST API design weight to 2.0× and the Docker weight to 0.0× increased Ayush's weighted score from 12 to 16 (as his REST API score of 30 was given double importance while his zero Docker score was excluded). This demonstrates the system's ability to let recruiters express role-specific priorities without re-invoking the LLM. The re-ranking is computed entirely client-side from cached JSON data in under 100ms, making it suitable for real-time exploration of what-if scenarios.

---

## 7. Conclusion

This project demonstrates that Large Language Models can effectively automate the first-pass screening of candidate resumes against structured job requirements. The CV Sorter tool successfully parses PDF resumes, extracts structured job descriptions via LLM, scores each candidate on a per-requirement basis with cited evidence, and produces a ranked output that is both human-readable and machine-parseable.

Cross-provider testing revealed that different LLMs produce consistent relative rankings while varying in absolute scores by a small margin (3–8 points). This suggests that LLM-based scoring is reliable for shortlisting purposes, though absolute score thresholds should not be treated as definitive. Evidence quality also varies by provider — GPT-4o produced more nuanced explanations that better distinguished between related and directly relevant experience.

The Streamlit dashboard adds significant practical value by allowing recruiters to interact with scoring results without technical knowledge. The interactive re-ranking feature addresses a key limitation of static LLM scoring: different hiring managers can apply their own skill prioritisation to the same set of results, enabling personalised shortlisting without additional API calls or costs. The CSV export and CLI filter flags provide additional flexibility for integrating results into existing recruitment workflows.

The architecture's pluggable design proved valuable in practice: adding the Groq provider required only a new client class and a config entry, with zero changes to the scoring, parsing, or output modules. The same pattern enables future extensions including PaddleOCR for scanned PDFs (already architecturally designed as a two-file change), RAG-based retrieval for large resume databases, and agentic self-correction loops for improved scoring accuracy.

The primary limitation of the current system is its reliance on LLM output consistency. While the retry mechanism handles formatting failures (malformed JSON), the system cannot verify the factual accuracy of the LLM's evidence citations against the actual resume text. Additionally, Streamlit UI testing was not included in the test suite as it requires specialised tooling beyond the project's scope. Future work could address these through a verification pass that cross-references cited evidence with parsed resume content, and the adoption of Streamlit's AppTest framework for UI regression testing.

In summary, CV Sorter validates the feasibility of using LLMs as structured evaluation tools in recruitment. The combination of automated scoring, transparent evidence, multi-format output, and interactive re-ranking provides a foundation that can be extended into a production-grade screening system with additional validation and scale optimisations.
