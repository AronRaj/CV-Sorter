# CV Sorting Using LLMs — Capstone Report

---

## Abstract

Manual resume screening is one of the most time-consuming tasks in recruitment. Hiring managers spend an average of six to eight seconds per resume, leading to inconsistent evaluations and qualified candidates being overlooked. This project presents CV Sorter, a Python-based tool that automates resume ranking using a multi-agent LLM pipeline. The system uses four specialised agents — a Shortlist agent (Ollama Llama 3.1, local), a JD extraction agent (Ollama Gemma 2, local), a Scorer agent (Claude Sonnet, API), and a Report agent (Claude Sonnet, API) — to screen, score, and synthesise candidate evaluations against a job description. Local models handle high-volume, low-stakes tasks (shortlisting and JD parsing) at zero API cost, while Claude handles quality-sensitive tasks (scoring and recruiter briefing) where output is directly visible to the hiring team. The system ingests a job description and a folder of PDF resumes, scores each resume on a 0–100 scale across every requirement with textual evidence citations, and produces both a ranked JSON output and a recruiter summary with hire recommendations, skill gap analysis, tailored interview questions, and red flags. A Streamlit dashboard provides interactive exploration with evidence drill-down and re-ranking via per-requirement weight sliders. The entire pipeline runs with a single command — `python main.py` — with no required arguments. Testing with three candidate resumes against a Senior Android Developer job description produced clear differentiation: the top candidate scored 78/100 with evidence-backed strengths in Jetpack Compose and team leadership, while the other two scored 58/100 each with appropriate identification of experience gaps. A test suite of 22 unit and integration tests validates the system without API calls. The project demonstrates that a multi-agent LLM architecture can produce structured, evidence-based resume evaluations suitable for augmenting — not replacing — human recruitment decisions, while optimising for both cost efficiency and output quality through task-specific model assignment.

---

## 1. Introduction

The hiring process in modern organisations begins with resume screening — a largely manual activity where recruiters evaluate dozens or hundreds of candidate resumes against a job description. This process is inherently subjective; different reviewers may weigh the same qualification differently, and fatigue from reading many similar documents introduces inconsistency. As organisations scale, the volume of applications makes thorough manual screening impractical.

Recent advances in Large Language Models (LLMs) have made it possible to automate structured text analysis tasks that previously required human judgement. LLMs can read a resume, identify relevant skills and experience, compare them against job requirements, and produce a scored evaluation — all within seconds and at a fraction of the cost of human review time.

This project, CV Sorter, explores the application of LLMs to resume ranking. It is a Python-based tool that takes a job description and a set of candidate resumes as input, runs them through a multi-agent pipeline, and produces a ranked list of candidates with per-requirement scores and textual evidence. The tool is designed around two key principles: transparency (every score must cite specific evidence from the resume) and task-specific model assignment (local models for speed-sensitive filtering tasks, API models for quality-sensitive scoring tasks). In addition to the CLI, the project provides a Streamlit web dashboard that enables recruiters to explore results interactively, filter candidates by score or skill, and re-rank candidates by adjusting requirement weights — all from cached scoring data with zero additional LLM calls.

The project was built incrementally over six days: Day 1 established the data models and parsing layer, Day 2 built the scoring engine and CLI pipeline, Day 3 added a test suite and edge-case hardening, Day 4 added the presentation and interactivity layer including CSV export, CLI filter flags, and the Streamlit dashboard, Day 5 migrated the architecture to a multi-agent pipeline using LangChain's BaseChatModel interface with four specialised agents (Shortlist, Scorer, Report, and a Supervisor coordinator), and Day 6 consolidated the provider set from five (Claude, OpenAI, Gemini, Groq, HuggingFace) to two (Claude for quality, Ollama for cost), establishing fixed model assignments as architectural decisions rather than runtime options. This report documents the design decisions, implementation, results, and lessons learned.

---

## 2. Problem Statement

Recruiting teams face a critical bottleneck at the resume screening stage. For a typical software engineering role, a company may receive 100–500 applications. A recruiter spending two minutes per resume would need over 16 hours of focused reading to screen 500 candidates — and this is before any interviews take place.

The core problems with manual screening are:

**Inconsistency.** Two recruiters evaluating the same resume against the same job description will often reach different conclusions. Fatigue, implicit bias, and subjective interpretation of qualifications all contribute to inconsistent outcomes.

**Lack of evidence.** When a recruiter shortlists or rejects a candidate, the reasoning is rarely documented in a structured way. This makes it difficult to audit decisions or provide constructive feedback to candidates.

**Scalability.** Manual screening does not scale with application volume. As companies grow and roles attract more applicants, the quality of screening degrades unless proportionally more recruiter time is allocated.

**Inflexible prioritisation.** Different hiring managers may prioritise different skills for the same role. A backend team lead may value PostgreSQL experience more heavily than Docker, while a DevOps-oriented manager may do the opposite. Static scoring systems cannot accommodate these preferences without re-running the entire evaluation.

**No cross-candidate synthesis.** Traditional screening evaluates candidates individually. Recruiters must mentally synthesise patterns across multiple resumes — common skill gaps, relative strengths, and red flags — without systematic support. This holistic view is critical for making informed hiring decisions but is the most cognitively demanding part of the process.

This project addresses these problems by delegating the structured comparison of resumes against job requirements to a multi-agent LLM pipeline, generating a cross-candidate recruiter briefing with recommendations and interview questions, and providing an interactive dashboard where recruiters can adjust skill weights and re-rank candidates without additional LLM calls. The system does not replace human decision-making; rather, it produces a ranked shortlist with explicit evidence for each score and a professional briefing that enables recruiters to focus their time on the most promising candidates.

---

## 3. Objectives

The objectives of this project are:

- **Build a zero-configuration CLI tool** that ranks candidate resumes against a job description with a single command (`python main.py`), producing ranked JSON output with per-requirement scores and evidence citations.

- **Implement a multi-agent pipeline** with task-specific model assignments — local Ollama models (Llama 3.1 for shortlisting, Gemma 2 for JD extraction) for fast, free screening, and Claude Sonnet for quality-sensitive scoring and report generation.

- **Extract structured data from PDFs** using pymupdf4llm to convert resume content into markdown that preserves formatting, tables, and multi-column layouts for optimal LLM comprehension.

- **Design an extensible PDF parsing layer** using the abstract factory pattern, with a pre-designed extension path for PaddleOCR to handle scanned/image-based PDFs as a two-file change.

- **Produce transparent, evidence-based evaluations** where every requirement score includes a textual evidence field citing specific content from the candidate's resume.

- **Implement a self-evaluation loop** in the Scorer agent that reviews its own evidence quality, flags weak scores, and re-scores once with a stricter prompt to improve evidence specificity.

- **Generate a recruiter summary** — a markdown briefing with hire recommendations, skill gap analysis across the candidate pool, tailored interview questions per candidate, and red flags or inconsistencies noted during scoring.

- **Build robust JSON parsing** with multi-attempt retry logic, code fence stripping, lenient parsing fallbacks, and defensive defaults to handle the inherent non-determinism of LLM output — especially from smaller local models.

- **Validate the system with a comprehensive test suite** (22 unit and integration tests) that runs without API calls using mock chat models and monkeypatched environment variables.

- **Provide a recruiter-facing user interface** with a Streamlit web dashboard featuring ranked candidate cards, expandable evidence drill-down, sidebar filters, and interactive re-ranking via per-requirement weight sliders.

- **Support multiple output formats and CLI filtering** including CSV export (--export-csv), minimum score threshold (--min-score), and verbose terminal table (--verbose) for recruiter-friendly data export and shortlisting.

---

## 4. Methodology

The system was implemented in Python 3.10+ following a two-layer architecture with strict separation of concerns. The domain layer (`src/core/`) contains pure business logic — data models, PDF parsing, scoring, JD extraction, and output formatting. The application layer (`src/agents/`) contains orchestration and agent behaviour — model construction, tool wrappers, individual agents, and the pipeline coordinator. Dependencies flow in one direction: the agent layer imports from the domain layer, but the domain layer never imports from the agent layer.

**Data Models.** All data structures are defined as Python dataclasses in a single `models.py` file: `Resume`, `JobDescription`, `RequirementScore`, and `CandidateScore`. This mirrors the data class pattern from Kotlin/Java and serves as the single source of truth for all data shapes exchanged between modules.

**PDF Parsing.** Resume PDFs are parsed using pymupdf4llm, which outputs markdown rather than plain text. This preserves structural information — tables become markdown tables, headings are marked, and multi-column layouts are correctly ordered. The parser is accessed through an abstract `PDFParser` interface with a factory method, allowing PaddleOCR to be plugged in for scanned PDFs without changing any calling code.

**LLM Integration.** All LLM access is through LangChain's `BaseChatModel` interface, which provides a unified `.invoke()` method across providers. The model factory module (`model_factory.py`) exposes three named constructor functions — `get_claude_model(config)`, `get_ollama_shortlist_model()`, and `get_ollama_jd_model()` — each returning a `BaseChatModel` configured for its specific task. This named-function pattern replaces the earlier generic dispatcher and makes model assignments explicit in the code. Prompt templates are stored in external `.txt` files and filled using Python string formatting, enabling prompt iteration without code changes.

**Multi-Agent Pipeline.** The system uses four agents coordinated by a Supervisor:

1. **ShortlistAgent** (Ollama Llama 3.1, local) — scans each resume against the job description and makes a binary PROCEED or SKIP decision. This fast, free first pass eliminates clearly unqualified candidates before any API calls are made.

2. **JD Extraction** (Ollama Gemma 2, local) — parses the raw job description text into a structured `JobDescription` object with fields for job title, required skills, nice-to-have skills, minimum experience, and responsibilities. This structured extraction enables per-requirement scoring.

3. **ScorerAgent** (Claude Sonnet, API) — deep-scores each shortlisted resume against every extracted requirement. Each score includes textual evidence citing specific content from the resume. After scoring, a self-evaluation loop reviews the evidence quality: if any requirement received a high score but weak evidence, the agent re-scores that candidate with a stricter prompt. This loop typically triggers for 1–2 candidates per run.

4. **ReportAgent** (Claude Sonnet, API) — reads all scored candidates together and synthesises a markdown recruiter briefing containing hire recommendations with reasoning, skill gap analysis across the candidate pool, tailored interview questions per candidate, and red flags or inconsistencies.

The Supervisor (`supervisor.py`) coordinates these agents in sequence: parse resumes, extract JD, shortlist, score, generate report, write output. It builds all models internally using the named factory functions — the caller passes only a `Config` object. Model assignments are fixed architectural decisions, not runtime options.

**JSON Resilience.** LLM-generated JSON is inherently fragile, especially from smaller local models. The system implements layered defences: (1) code fence stripping removes markdown formatting that local models sometimes wrap around JSON responses, (2) defensive `.get()` calls with default values handle missing keys without crashing, (3) a `_derive_fit_label()` fallback computes fit labels from scores when the LLM omits them, (4) a retry loop sends a repair prompt if JSON parsing fails, including the failed response for context, and (5) a lenient parser as a last resort extracts whatever valid JSON parts exist, ensuring the pipeline completes even with severely malformed LLM responses. The prompt template itself includes explicit instructions to avoid unescaped double quotes and multi-line strings inside JSON values — rules that significantly improved reliability with Ollama models.

**Output Formats.** The system produces three output formats: (1) a structured JSON file containing the full ranked list with all per-requirement scores and evidence, (2) a flat CSV file suitable for import into Excel or Google Sheets, and (3) a colour-coded Rich terminal table for quick command-line review. A fourth output — the recruiter summary markdown file — is generated by the Report agent. The JSON output always contains all candidates unfiltered; the CSV and terminal table respect any --min-score flag provided at the command line.

**Streamlit Dashboard.** A Streamlit web application (`streamlit_app.py`) provides a recruiter-facing interface that reads the JSON output produced by the pipeline. The dashboard displays ranked candidate cards with expandable evidence drill-down per requirement, using colour-coded score badges (green for 80+, blue for 60–79, orange for 40–59, red for 0–39). Sidebar controls provide a minimum score slider and a skill filter text input for live filtering. An interactive re-ranking section allows recruiters to adjust per-requirement weight sliders (0.0x to 2.0x) and re-sort candidates using a weighted score formula — all computed client-side from cached JSON data with zero additional LLM calls. This separation ensures that the dashboard is instant, free to use, and does not depend on API availability.

**Re-ranking Algorithm.** The interactive re-ranking computes a weighted normalised score for each candidate: for each requirement, the raw score is multiplied by the slider weight, and the results are summed and divided by the total weight. Setting a weight to 2.0x doubles that requirement's contribution; setting it to 0.0x removes it entirely. Fit labels are re-derived from the new weighted score using the same bands as the scorer (80–100 Strong match, 60–79 Good match, 40–59 Partial match, 0–39 Weak match). This allows recruiters to prioritise their most important requirements without re-running the LLM.

**Testing.** The test suite uses `MockChatModel` classes that implement the LangChain `BaseChatModel` `invoke()` interface, returning hardcoded JSON strings wrapped in message objects. Tests use pytest monkeypatch for environment isolation, `unittest.mock.patch` for avoiding real model construction in Supervisor tests, and a synthetic PDF for parser tests. All 22 tests run in under one second with zero API calls.

---

## 5. Model Selection Rationale

A key architectural decision in CV Sorter is the assignment of specific models to specific tasks based on the cost-quality tradeoff each task requires. Rather than using a single model for all operations, the system uses four distinct model assignments:

| Task | Model | Runs on | API Cost | Reasoning |
|---|---|---|---|---|
| Shortlisting | Llama 3.1 8B | Ollama (local) | Free | Binary PROCEED/SKIP decision. Speed matters more than nuance — the agent processes every resume, so it must be fast. A local model runs in 2–5 seconds per resume with zero API cost. False negatives are acceptable because the system falls back to scoring all candidates if none are shortlisted. |
| JD Extraction | Gemma 2 9B | Ollama (local) | Free | Structured field extraction from a job description. The task requires following instructions to extract named fields (job title, required skills, responsibilities) from semi-structured text — a task where instruction-following ability matters more than reasoning depth. Gemma 2's strength in instruction-following makes it well-suited, and running locally eliminates API latency for this blocking step. |
| Scoring | Claude Sonnet 4.5 | Anthropic API | Paid | Evidence-based scoring across 30+ requirements per resume. This task requires long-document reasoning (reading both a resume and a detailed JD), calibrated numerical scoring, and specific evidence citation. Claude Sonnet's strength in careful, well-calibrated analysis makes it the right choice for a task where output quality is directly visible to recruiters. |
| Report Writing | Claude Sonnet 4.5 | Anthropic API | Paid | Cross-candidate synthesis into a professional recruiter briefing. The report agent reads all scored candidates together and produces recommendations, skill gap analysis, interview questions, and red flags. This requires multi-document reasoning and professional prose quality — capabilities where Claude excels. |

The design principle is straightforward: local models for high-volume, low-stakes tasks; API models for low-volume, high-stakes tasks. Shortlisting processes every resume (high volume) but only makes a binary decision (low stakes — mistakes are recoverable). Scoring and report writing process fewer candidates (only shortlisted ones) but produce the outputs recruiters directly read and act on (high stakes — quality must be high).

This approach was arrived at through iteration. The initial system (Days 1–4) supported five providers (Claude, OpenAI, Gemini, Groq, HuggingFace) as runtime options, which added configuration complexity without functional benefit. On Day 6, the provider set was consolidated to two — Claude and Ollama — with fixed assignments. This made the system simpler to configure (no provider selection flags), cheaper to run (two of four tasks are free), and more reliable (fewer failure modes from provider misconfiguration).

---

## 6. User Interface

The system provides two interfaces: a command-line interface (CLI) for batch processing and a Streamlit web dashboard for interactive exploration.

### 6.1 Command-Line Interface

The CLI is the primary interface for running the pipeline. The default invocation requires no arguments:

```
python main.py
```

This reads `job_description.txt` from the project root, scans PDFs in `src/resumes/`, and writes output to `src/results/`. A startup banner displays the model assignments:

```
CV Sorter — Multi-Agent Recruitment Pipeline
  Shortlisting : Ollama llama3.1:latest (local)
  JD analysis  : Ollama gemma2:latest (local)
  Scoring      : Claude claude-sonnet-4-5 (API)
  Report       : Claude claude-sonnet-4-5 (API)
```

Optional flags extend functionality for power users:

- **--export-csv PATH** writes a flat CSV file alongside the JSON output, suitable for import into Excel or Google Sheets.

- **--min-score N** filters the terminal table and CSV export to show only candidates with overall scores at or above the specified threshold. The JSON file is unaffected.

- **--verbose** prints a colour-coded ranked table to the terminal after the pipeline completes.

These flags are applied after the pipeline completes, ensuring the full unfiltered results are always preserved in the JSON output for later analysis or dashboard use.

### 6.2 Streamlit Dashboard

The Streamlit dashboard (`streamlit_app.py`) provides a web-based interface for recruiters to explore scoring results interactively. It loads a ranked JSON file produced by the CLI and displays three main sections:

**Candidate Cards.** Each candidate is shown in an expandable card displaying their rank, filename, overall score, and fit label. Expanding a card reveals the LLM's explanation and a per-requirement breakdown with colour-coded score badges and evidence citations.

**Sidebar Filters.** A minimum score slider (0–100) and a skill filter text input allow recruiters to narrow the displayed candidates in real time. Filters update live as controls are adjusted — no button click required.

**Interactive Re-ranking.** Below the candidate cards, a set of sliders (one per requirement, range 0.0x to 2.0x, default 1.0x) allows recruiters to adjust the importance of each requirement. Clicking the Re-rank button recomputes weighted scores and re-sorts candidates instantly. Rank changes are annotated (e.g., "was #2"). This feature enables different hiring managers to apply their own prioritisation without re-running the LLM.

---

## 7. Results and Analysis

The system was tested using three candidate resumes — Aron (Senior Android developer, 13+ years experience), Ayush Kushwaha (Android developer, 3 years experience), and Saumitra (Android developer at Samsung, 4.5 years experience) — against a Senior Android Developer job description requiring 10+ years of experience, deep expertise in Kotlin, Jetpack Compose, MVVM/MVI, performance profiling, testing frameworks, and technical leadership.

### Multi-Agent Pipeline Results

| Rank | Candidate | Score | Fit Label | Key Strength | Key Gap |
|------|-----------|-------|-----------|-------------|---------|
| 1 | Aron | 78/100 | Good match | 13+ years, led Compose migration at Walmart, mentored teams | MVI, GraphQL, Hilt/Dagger not evidenced |
| 2 | Ayush | 58/100 | Partial match | Strong Kotlin/Compose skills, active Compose migration lead | Only 3 years experience, no leadership evidence |
| 3 | Saumitra | 58/100 | Partial match | Samsung R&D background, MVI knowledge | No Compose, no testing frameworks, no CI/CD |

### Key Observations

**Observation 1: The shortlist agent correctly passed all three candidates.** All three candidates had Android development backgrounds relevant to the Android Developer JD, so the Ollama-based shortlist agent correctly issued PROCEED decisions for all three. In a larger candidate pool with unrelated resumes (e.g., a data scientist applying for an Android role), the shortlist agent would filter those out before any API calls are made. The agent's reasoning was transparent — for Aron it noted "strong presence of required skills, including MVVM and MVI, and extensive experience," while for Saumitra it noted the candidate "lacks MVI skill and significant seniority" but still proceeded given partial matches.

**Observation 2: The self-evaluation loop improved evidence quality.** For Aron (the top candidate), the scorer's initial pass flagged 13 requirements with weak evidence, including Jetpack Navigation, WorkManager, Hilt/Dagger, and several leadership responsibilities. The self-evaluation loop triggered a re-scoring pass that produced more specific evidence citations for those requirements. For example, after re-scoring, the evidence for "Lead code reviews, enforce architectural standards" was updated to reference his Scrum Master role and team mentorship rather than generic inferences. The other two candidates' initial scores passed self-evaluation without triggering re-scoring, indicating the loop activates selectively rather than uniformly.

**Observation 3: Score differentiation aligns with experience level.** The 20-point gap between Aron (78) and the other two candidates (both 58) directly corresponds to the experience gap: Aron has 13+ years with leadership responsibilities, while Ayush has 3 years and Saumitra has 4.5 years as individual contributors. The scorer correctly identified that technical skills alone are insufficient for a senior role — both Ayush and Saumitra scored well on core skills (Kotlin 80–90, MVVM 75–85) but scored poorly on leadership requirements (mentoring 10–30, architecture ownership 20–30, strategic planning 20). This differentiation is more valuable to a recruiter than a flat "match/no match" classification.

**Observation 4: The recruiter summary provides actionable intelligence beyond individual scores.** The Report agent's output included four sections that go beyond what per-candidate scoring can provide:

- **Hire recommendation**: "Interview Aron first. He is the only candidate who meets the senior-level requirements for this role."
- **Skill gap analysis**: Identified that Hilt/Dagger, advanced testing frameworks (Mockito, Robolectric), and GraphQL were absent across all three candidates, suggesting these may be unrealistic requirements for the available talent pool.
- **Interview questions**: Generated tailored questions per candidate — for Aron, probing his MVI knowledge gap; for Ayush, assessing leadership potential; for Saumitra, verifying claimed expertise depth.
- **Red flags**: Flagged Saumitra's resume as potentially "keyword-optimised rather than reflecting deep, demonstrable expertise" based on the pattern of listing many technologies without project-specific evidence.

### Per-Requirement Score Breakdown (Aron — Top Candidate)

| Requirement | Score | Evidence Summary |
|---|---|---|
| Android development | 95 | 13+ years across mobile, TV, wearables at Walmart, Nagra Vision, others |
| Kotlin | 90 | Led first production Compose feature (requires Kotlin) at Sam's Club |
| Jetpack Compose | 95 | Spearheaded Digital Cakes feature, 50% code reduction, 25% performance boost |
| MVVM | 90 | Explicitly listed, profile summary references MVVM and SOLID principles |
| Performance optimisation | 90 | Optimised apps for 2M+ users, monitoring dashboards, zero production incidents |
| Mentoring teams | 85 | Scrum Master, mentored junior developers at Nagra Vision |
| CI/CD | 80 | CI/CD listed as core skill, Scrum Master role |
| MVI | 70 | Listed as skill but no project examples — appropriately scored lower |
| Hilt/Dagger | 50 | Not mentioned — lowest-scored requirement, reflecting genuine gap |

The scoring pattern demonstrates appropriate calibration: skills with concrete project evidence (Compose at 95, Kotlin at 90) score significantly higher than skills merely listed without evidence (MVI at 70, Hilt/Dagger at 50). This evidence-quality sensitivity is a direct result of the prompt template instructing the LLM to score based on demonstrated experience, not keyword presence.

### Pipeline Performance

The complete pipeline processed three resumes in approximately 260 seconds. The local Ollama models (shortlisting and JD extraction) completed in under 30 seconds total. The majority of the time was spent on Claude API calls for scoring (3 candidates x 2 passes for self-evaluation) and report generation. For a larger batch of 20 resumes where the shortlist agent filters half, the estimated cost would be approximately 10 Claude API calls (10 scoring + 1 report) rather than 20, representing a 50% cost reduction from the pre-agent architecture.

---

## 8. Conclusion

This project demonstrates that a multi-agent LLM architecture can effectively automate the first-pass screening of candidate resumes against structured job requirements. The CV Sorter tool successfully parses PDF resumes, extracts structured job descriptions via LLM, shortlists candidates using a local model, deep-scores each shortlisted candidate with per-requirement evidence using Claude, and generates a cross-candidate recruiter briefing — all from a single `python main.py` command.

The task-specific model assignment proved to be the most impactful design decision. By using free local models for shortlisting and JD extraction, and reserving the paid API model for scoring and report writing, the system achieves a practical cost-quality balance: the recruiter-facing outputs (scores and briefing) are produced by the highest-quality model available, while the filtering and parsing tasks that happen behind the scenes run at zero cost. This architecture scales naturally — doubling the number of resumes doubles the (free) shortlisting cost but not the (paid) scoring cost, since only shortlisted candidates reach the scorer.

The self-evaluation loop in the Scorer agent addresses a key limitation of single-pass LLM scoring: the tendency to assign high scores without specific evidence. By reviewing its own output and re-scoring when evidence is weak, the agent produces more defensible evaluations. In testing, this loop activated for 1 of 3 candidates, improving evidence specificity for 13 requirements without changing the overall ranking.

The recruiter summary — generated by the Report agent reading all scored candidates together — provides value that per-candidate scoring alone cannot. The cross-candidate skill gap analysis, for example, revealed that Hilt/Dagger and advanced testing frameworks were absent across the entire candidate pool, suggesting the JD requirements may need adjustment. This holistic insight is the kind of pattern recognition that human recruiters do manually but rarely document systematically.

The Streamlit dashboard adds significant practical value by allowing recruiters to interact with scoring results without technical knowledge. The interactive re-ranking feature addresses a key limitation of static LLM scoring: different hiring managers can apply their own skill prioritisation to the same set of results, enabling personalised shortlisting without additional API calls or costs.

The primary limitation of the current system is its reliance on LLM output consistency. While the multi-layer JSON resilience mechanisms (retry, code fence stripping, lenient parsing) handle formatting failures robustly, the system cannot verify the factual accuracy of the LLM's evidence citations against the actual resume text. A second limitation is that the self-evaluation loop adds latency — approximately 15–20 seconds per candidate for the additional Claude API call. Future work could address factual accuracy through a verification pass that cross-references cited evidence with parsed resume content, and reduce self-evaluation latency through parallel API calls.

In summary, CV Sorter validates the feasibility of using a multi-agent LLM pipeline as a structured evaluation tool in recruitment. The combination of local model screening, API model scoring, self-evaluation, cross-candidate synthesis, and interactive re-ranking provides a foundation that can be extended into a production-grade screening system with additional validation, scale optimisations, and the conversational AI capabilities described in the next section.

---

## 9. Possible Improvements and Future Work

### AI Recruiter Agent

The most impactful extension would be a conversational AI recruiter agent that conducts preliminary screening interviews via chat. After the current pipeline ranks candidates and identifies their strengths and gaps, the AI recruiter could:

- **Conduct structured interviews via chat or voice** — asking candidates to elaborate on resume claims, explain career transitions, and provide context for skill gaps identified during scoring.
- **Ask follow-up questions about specific gaps** — for example, if a candidate scored low on testing frameworks, the agent could ask "Can you describe your testing approach and which frameworks you have used?" to gather evidence not present in the resume.
- **Assess communication skills** — by evaluating response clarity, technical depth, and professionalism in the candidate's answers, providing a dimension that resume analysis alone cannot capture.
- **Generate interview readiness scores** — combining the resume-based score with the conversation-based assessment to produce a more holistic pre-screening evaluation.
- **Provide candidates with immediate feedback** — explaining what the role requires and how their profile aligns, improving the candidate experience even for those who are not shortlisted.

This would extend the pipeline from document analysis to interactive candidate engagement, bridging the gap between automated screening and the human interview. The existing architecture supports this naturally — the AI recruiter would be a fifth agent in the Supervisor's pipeline, receiving the scored CandidateScore objects and the recruiter summary as context for its conversations.

### Additional Improvements

- **PaddleOCR for scanned PDFs.** The PDF parsing layer already has an abstract interface with a factory method and a stub implementation for PaddleOCR. Enabling it is a two-file change: implement the `PaddleOCRParser` class and add the factory case in `PDFParser.build()`. This would extend the system to handle scanned or image-based resumes that pymupdf4llm cannot extract text from.

- **RAG-based retrieval for large resume databases.** For organisations with thousands of historical resumes, a retrieval-augmented generation (RAG) approach could index resumes in a vector database and retrieve the most relevant candidates before running the scoring pipeline. This would reduce the number of resumes processed from thousands to a targeted subset.

- **LangSmith tracing for pipeline observability.** The system already has LangSmith integration points (environment variable configuration). Enabling tracing would provide detailed visibility into each agent's LLM calls — prompt content, token usage, response latency, and quality metrics — enabling systematic prompt optimisation and cost monitoring.

- **Groq as a fast API alternative.** The architecture supports adding new providers through a single factory function. Groq's OpenAI-compatible API with Llama models could serve as a fast, cost-effective alternative to Claude for scoring in scenarios where latency is more important than maximum quality — for example, processing hundreds of candidates in a high-volume hiring event.

- **Evidence verification pass.** A post-scoring step that cross-references each evidence citation with the actual parsed resume text, flagging cases where the LLM's cited evidence does not appear in the source document. This would address the factual accuracy limitation noted in the conclusion.

- **Streamlit UI testing.** Adopting Streamlit's AppTest framework for automated UI regression testing would ensure dashboard functionality is preserved as the system evolves. This was out of scope for the capstone but would be essential for production deployment.
