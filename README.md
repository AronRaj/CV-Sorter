# CV Sorter

A multi-agent LLM pipeline that ranks candidate resumes against a job description. Uses local models for fast screening (free) and Claude for deep analysis (API). Produces a ranked results file and a recruiter briefing with interview questions.

## How it works

The pipeline runs four agents in sequence. First, the **Shortlist agent** (local Llama 3.1 8B via Ollama) scans every resume and makes a fast binary decision — proceed to scoring or skip — at zero API cost. In parallel, the **JD extraction agent** (local Gemma 2 9B) parses the job description into structured requirements. Then, the **Scorer agent** (Claude Sonnet) deep-scores each shortlisted candidate against every requirement with specific evidence, running a self-evaluation loop that flags weak evidence for re-scoring. Finally, the **Report agent** (Claude Sonnet) reads all scored candidates together and writes a recruiter briefing with hire recommendations, skill gap analysis, tailored interview questions, and red flags.

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install Ollama and pull local models

Download and install Ollama from [https://ollama.com/download](https://ollama.com/download), then pull the two models used by the pipeline:

```bash
ollama pull llama3.1       # shortlist agent (~4.9 GB, defaults to :latest = 8B)
ollama pull gemma2         # JD extraction agent (~5.4 GB, defaults to :latest = 9B)
```

> **Low RAM (< 16 GB)?** Use smaller alternatives:
>
> ```bash
> ollama pull llama3.2:3b    # instead of llama3.1
> ollama pull gemma2:2b      # instead of gemma2
> ```
>
> Then update `OLLAMA_SHORTLIST_MODEL` and `OLLAMA_JD_MODEL` in `src/core/config.py` to match.

### 3. Add your API key

```bash
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY
# Get a key from https://console.anthropic.com
```

### 4. Add your files

```
src/resumes/          ← drop candidate PDF resumes here
job_description.txt   ← write your job description here (plain text)
```

The `job_description.txt` format is plain English paragraphs describing the role, required skills, experience level, and responsibilities. No special format needed — the JD agent extracts the structure automatically.

## Run

```bash
python main.py
```

That is the entire command. No arguments required.

The pipeline runs automatically and produces two output files:

```
src/results/ranked_output.json     — ranked candidates with per-requirement scores
src/results/recruiter_summary.md   — recruiter briefing with interview questions
```

## Output files

### src/results/ranked_output.json

```json
{
  "job_title": "Senior Android Developer",
  "model": "claude-sonnet-4-5",
  "ranked_at": "2026-03-22T14:30:00",
  "total_candidates": 3,
  "candidates": [
    {
      "rank": 1,
      "filename": "john_doe.pdf",
      "overall_score": 87,
      "fit_label": "Strong match",
      "explanation": "Strong Android experience with 8 years of Kotlin...",
      "requirement_scores": [
        { "requirement": "Kotlin / Jetpack Compose", "score": 90, "evidence": "Led Compose migration at..." },
        { "requirement": "CI/CD pipelines", "score": 75, "evidence": "Set up GitHub Actions for..." }
      ]
    }
  ]
}
```

Each candidate includes an `overall_score` (0–100), a `fit_label` (Strong match / Moderate match / Weak match), a free-text `explanation`, and per-requirement breakdowns with evidence quotes from the resume.

### src/results/recruiter_summary.md

A free-form recruiter briefing written by the Report agent. Includes:

- **Hire recommendation** with reasoning for each candidate
- **Skill gap analysis** across the candidate pool
- **Tailored interview questions** per candidate based on their specific profile
- **Red flags** or inconsistencies noted during scoring

## Optional flags

All CLI arguments are optional. The primary workflow is `python main.py` with no flags.

| Flag | Default | Description |
|---|---|---|
| `--jd PATH` | `job_description.txt` | Path to job description text file |
| `--resumes PATH` | `src/resumes/` | Folder containing PDF resumes |
| `--output PATH` | `src/results/ranked_output.json` | Path for ranked JSON output |
| `--summary PATH` | `src/results/recruiter_summary.md` | Path for recruiter summary markdown |
| `--export-csv PATH` | — | Also export results as CSV to this path |
| `--min-score N` | `0` | Only show candidates scoring >= N in terminal output |
| `--verbose` | off | Print ranked table to terminal after run |

The JSON output always includes **all** candidates. `--min-score` affects terminal display and CSV export only.

## Dashboard

```bash
streamlit run streamlit_app.py
```

Opens a browser dashboard at [http://localhost:8501](http://localhost:8501) where you can:

- Load any results JSON file
- Browse candidates with evidence drill-down
- Adjust requirement weights and re-rank without re-running the pipeline
- Filter by minimum score or specific skill

## Model choices

| Task | Model | Why |
|---|---|---|
| JD extraction | Ollama Gemma 2 (local) | Structured field extraction — no reasoning needed |
| Shortlisting | Ollama Llama 3.1 (local) | Binary PROCEED/SKIP decision — speed over quality |
| Deep scoring | Claude Sonnet (API) | Long-document reasoning with calibrated evidence |
| Report writing | Claude Sonnet (API) | Professional prose and multi-document synthesis |

Local models (Ollama) run at zero API cost and handle extraction and filtering tasks that do not require nuanced reasoning. Claude is reserved for tasks where output quality is directly visible to the recruiter — scoring and the recruiter briefing.

## Project structure

```
CV-Sorter/
├── main.py                          entry point — run this
├── streamlit_app.py                 recruiter dashboard
├── job_description.txt              write your JD here
├── requirements.txt
├── .env                             API keys (never commit)
│
├── src/
│   ├── core/                        domain layer — business logic
│   │   ├── models.py                dataclasses: Resume, JobDescription, CandidateScore
│   │   ├── config.py                .env → typed Config dataclass + model constants
│   │   ├── parser_engine.py         PDF → Resume (PyMuPDF + PaddleOCR stub)
│   │   ├── jd_parser_engine.py      JD text → JobDescription via LLM
│   │   ├── scorer_engine.py         Resume + JD → CandidateScore via LLM
│   │   └── output_engine.py         results → JSON, CSV, Rich table
│   │
│   ├── agents/                      agent layer — orchestration
│   │   ├── model_factory.py         named model constructors (Claude + Ollama)
│   │   ├── tools.py                 @tool wrappers around src/core/ modules
│   │   ├── shortlist_agent.py       fast-pass filter (Ollama)
│   │   ├── scorer_agent.py          deep scoring with self-eval loop (Claude)
│   │   ├── report_agent.py          recruiter synthesis (Claude)
│   │   └── supervisor.py            pipeline coordinator
│   │
│   ├── prompts/                     prompt templates (.txt, filled with .format())
│   │   ├── score_candidate.txt
│   │   ├── extract_jd.txt
│   │   ├── quick_scan.txt
│   │   ├── self_eval.txt
│   │   └── report.txt
│   │
│   ├── resumes/                     drop candidate PDFs here
│   │
│   └── results/                     output directory (gitignored)
│
├── tests/                           unit tests (pytest)
│   ├── test_models.py
│   ├── test_config.py
│   ├── test_parser.py
│   ├── test_scorer.py
│   └── test_supervisor.py
```

## Tests

```bash
python -m pytest tests/ -v
```

## Requirements

- Python 3.10+
- Ollama installed and running ([https://ollama.com](https://ollama.com))
- Anthropic API key ([https://console.anthropic.com](https://console.anthropic.com))
- 10–16 GB RAM recommended for local models
- PDF resumes must be text-based (not scanned images)

## Docs

Architecture and module contracts: [ARCHITECTURE.md](ARCHITECTURE.md).
