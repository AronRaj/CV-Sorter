# CV-Sorter

CLI tool that ranks candidate **PDF resumes** against a **plain-text job description** using an LLM. It extracts structured requirements from the JD, scores each resume, sorts by fit, and writes **JSON** (full results). Optional **terminal table**, **CSV export**, and filters for display/export only.

Capstone project for the IIT-K AI course.

## Requirements

- Python 3.10+
- At least one API key: Anthropic, OpenAI, Google Gemini, or Groq (see `.env.example`)

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env — set keys and DEFAULT_PROVIDER
```

## Usage

```bash
python main.py --jd job_description.txt --resumes ./resumes
```

Common flags:

| Flag | Purpose |
|------|--------|
| `--provider` | `claude`, `openai`, `gemini`, or `groq` (overrides `.env`) |
| `--output` | JSON output path (default: `OUTPUT_PATH` in `.env`) |
| `--verbose` | Print a ranked table to the terminal |
| `--export-csv PATH` | Also write CSV |
| `--min-score N` | Limit table/CSV to scores ≥ N |
| `--filter-skills "A,B"` | Limit table/CSV to candidates strong on all listed skills |

JSON output always includes **all** candidates; `--min-score` and `--filter-skills` affect **display and CSV only**.

## Tests

```bash
python -m pytest tests/
```

## Docs

Design and module contracts: [ARCHITECTURE.md](ARCHITECTURE.md).
