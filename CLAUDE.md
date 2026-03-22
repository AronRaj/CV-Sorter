# Claude Code Instructions

This file gives Claude Code the context it needs to work on this project effectively.
Read ARCHITECTURE.md first for module contracts and design decisions.

---

## Project Summary

Multi-agent LLM pipeline that ranks candidate resumes against a job description.
Uses a local model for fast screening and an API model for deep scoring.
Language: Python 3.10+. Streamlit dashboard for interactive exploration. Runs from the terminal.

---

## Code Style Rules

These apply to every file you write or edit in this project.

### General
- Use explicit type hints on every function signature — treat them like Kotlin type declarations
- Use dataclasses for all data structures (defined in `src/core/models.py` — never define new ones elsewhere)
- Break logic into small, named methods — prefer 10–20 line functions over long ones
- Use named arguments when calling functions with 3+ parameters
- Constants go at the top of the file in UPPER_SNAKE_CASE
- No one-liners using walrus operator or complex list comprehensions — keep it readable

### Naming
- Classes: `PascalCase` (same as Java/Kotlin)
- Functions and variables: `snake_case`
- Private methods: `_leading_underscore`
- Constants: `UPPER_SNAKE_CASE`
- Files: `snake_case.py`

### Imports
- Standard library first, then third-party, then local — separated by blank lines
- Use absolute imports: `from src.core.models import Resume` not relative `from .models import Resume`

### Error handling
- Use specific exception types, not bare `except:`
- Log warnings with `logging.warning()` before skipping a file — never silently swallow errors
- Raise `ValueError` for invalid inputs, `RuntimeError` for unrecoverable states

### Comments
- Docstrings on every class and public method
- Inline comments only for non-obvious logic — not for obvious things like `# increment counter`

---

## Module Responsibilities (quick reference)

### Entry point

| File              | What it does                                               | What it must NOT do              |
|-------------------|------------------------------------------------------------|----------------------------------|
| `main.py`         | Parse CLI args, build models, call Supervisor              | Contain business logic           |

### Domain layer (`src/core/`)

| File                          | What it does                                               | What it must NOT do              |
|-------------------------------|------------------------------------------------------------|----------------------------------|
| `src/core/models.py`         | Define dataclasses only                                    | Import from other src/ files     |
| `src/core/config.py`         | Load .env, return Config dataclass                         | Call any LLM or parse files      |
| `src/core/parser_engine.py`  | `PDFParser` ABC + `PyMuPDFParser` + `PaddleOCRParser` stub. `ResumeParser` wraps the active parser. | Call the LLM |
| `src/core/jd_parser_engine.py`| JD text → JobDescription via BaseChatModel                | Parse PDFs                       |
| `src/core/scorer_engine.py`  | Resume + JD → CandidateScore via BaseChatModel + prompt    | Write output files               |
| `src/core/output_engine.py`  | CandidateScore list → JSON, CSV, Rich table                | Call the LLM                     |

### Application layer (`src/agents/`)

| File                            | What it does                                        | What it must NOT do              |
|---------------------------------|-----------------------------------------------------|----------------------------------|
| `src/agents/model_factory.py`  | Named model constructors: `get_claude_model()`, `get_ollama_shortlist_model()`, `get_ollama_jd_model()` | Know about resumes or scoring |
| `src/agents/tools.py`          | `@tool` wrappers around src/core/ modules           | Contain business logic           |
| `src/agents/shortlist_agent.py`| Fast-pass filter using local model (Ollama)         | Deep-score candidates            |
| `src/agents/scorer_agent.py`   | Deep scoring with self-evaluation loop              | Filter candidates                |
| `src/agents/report_agent.py`   | Cross-candidate recruiter synthesis                 | Score or filter candidates       |
| `src/agents/supervisor.py`     | Sequential coordinator: parse → shortlist → score → report | Contain scoring or parsing logic |

---

## Working with Prompts

Prompt templates are in `src/prompts/*.txt`. To fill a template:

```python
template = Path("src/prompts/score_candidate.txt").read_text()
prompt = template.format(
    job_title=jd.job_title,
    required_skills=", ".join(jd.required_skills),
    resume_text=resume.raw_text
)
```

When editing a prompt:
- Always end the prompt with an instruction to return only valid JSON with no extra text
- Test the new prompt against at least 2 sample resumes in `tests/sample_resumes/` before committing
- If the LLM starts returning non-JSON responses, the problem is almost always in the prompt, not the parser

---

## Common Tasks

### Add a new LLM provider
See `ARCHITECTURE.md` → "Adding a New Provider". Changes needed in `src/agents/model_factory.py`, `src/core/config.py`, `main.py`, and `README.md` only.

### Enable PaddleOCR (post-MVP)
See `ARCHITECTURE.md` → "Plugging in PaddleOCR". Changes needed in `src/core/parser_engine.py` only (implement `PaddleOCRParser` + add factory case). Set `PDF_PARSER=paddle` in `.env`. No other files change.

### Change the scoring criteria
Edit `src/prompts/score_candidate.txt`. No Python changes needed unless you're adding new fields to `CandidateScore` in `src/core/models.py`.

### Add a new output format (e.g. CSV)
Add a new method to `src/core/output_engine.py`. Add a `--format` CLI arg to `main.py`. Do not change any other file.

### Debug a bad score
1. Run with `--verbose` to see the raw LLM response
2. Check `src/prompts/score_candidate.txt` for ambiguous instructions
3. Check that the resume PDF is parsing correctly (`src/core/parser_engine.py`)
4. Check that the JD extraction is picking up the right requirements (`src/core/jd_parser_engine.py`)

---

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run a specific test
python -m pytest tests/test_scorer.py -v
```

Sample resumes for testing are in `tests/sample_resumes/`. These are synthetic — do not replace them with real candidate data.

---

## Environment Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add at least one API key
```

---

## Files to Never Modify Without Reading ARCHITECTURE.md First

- `src/core/models.py` — changing a dataclass field breaks all callers
- `src/agents/model_factory.py` — the provider interface contract must stay stable
- `src/agents/supervisor.py` — the execution order and agent wiring matters

---

## Dependency Notes (for Kotlin/Java developer context)

| Python concept     | Kotlin equivalent           |
|--------------------|-----------------------------|
| `@dataclass`       | `data class`                |
| `ABC` + `@abstractmethod` | `interface` / `abstract class` |
| `list[str]`        | `List<String>`              |
| `str \| None`      | `String?`                   |
| `logging.warning()`| `Log.w()`                   |
| `Path("file").read_text()` | `File("file").readText()` |
| `argparse`         | Command-line argument parser (no direct equivalent — similar to kotlinx-cli) |
| `python-dotenv`    | Reading from a `.env` file (no built-in equivalent in Kotlin) |
