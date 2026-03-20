# Claude Code Instructions

This file gives Claude Code the context it needs to work on this project effectively.
Read ARCHITECTURE.md first for module contracts and design decisions.

---

## Project Summary

CLI tool that ranks candidate resumes against a job description using LLMs.
Language: Python 3.10+. No UI. No database. Runs entirely from the terminal.

---

## Code Style Rules

These apply to every file you write or edit in this project.

### General
- Use explicit type hints on every function signature — treat them like Kotlin type declarations
- Use dataclasses for all data structures (defined in `src/models.py` — never define new ones elsewhere)
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
- Use absolute imports: `from src.models import Resume` not relative `from .models import Resume`

### Error handling
- Use specific exception types, not bare `except:`
- Log warnings with `logging.warning()` before skipping a file — never silently swallow errors
- Raise `ValueError` for invalid inputs, `RuntimeError` for unrecoverable states

### Comments
- Docstrings on every class and public method
- Inline comments only for non-obvious logic — not for obvious things like `# increment counter`

---

## Module Responsibilities (quick reference)

| File              | What it does                                               | What it must NOT do              |
|-------------------|------------------------------------------------------------|----------------------------------|
| `main.py`         | Parse CLI args, build Config, call Pipeline                | Contain business logic           |
| `src/pipeline.py` | Coordinate module calls in order                           | Contain business logic           |
| `src/models.py`   | Define dataclasses only                                    | Import from other src/ files     |
| `src/config.py`   | Load .env, return Config dataclass                         | Call any LLM or parse files      |
| `src/llm_client.py` | Abstract interface + 3 provider implementations         | Know about resumes or scoring    |
| `src/parser.py`   | `PDFParser` abstract base + `PyMuPDFParser` (MVP) + `PaddleOCRParser` (post-MVP stub). `ResumeParser` wraps the active parser. | Call the LLM |
| `src/jd_parser.py`| JD text → JobDescription dataclass via LLM                | Parse PDFs                       |
| `src/scorer.py`   | Resume + JD → CandidateScore via LLM + prompt             | Write output files               |
| `src/output.py`   | CandidateScore list → JSON file                            | Call the LLM                     |

---

## Working with Prompts

Prompt templates are in `prompts/*.txt`. To fill a template:

```python
template = Path("prompts/score_candidate.txt").read_text()
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
See `ARCHITECTURE.md` → "Adding a New Provider". Changes needed in `llm_client.py`, `config.py`, `main.py`, and `README.md` only.

### Enable PaddleOCR (post-MVP)
See `ARCHITECTURE.md` → "Plugging in PaddleOCR". Changes needed in `parser.py` only (implement `PaddleOCRParser` + add factory case). Set `PDF_PARSER=paddle` in `.env`. No other files change.

### Change the scoring criteria
Edit `prompts/score_candidate.txt`. No Python changes needed unless you're adding new fields to `CandidateScore` in `models.py`.

### Add a new output format (e.g. CSV)
Add a new method to `src/output.py`. Add a `--format` CLI arg to `main.py`. Do not change any other file.

### Debug a bad score
1. Run with `--verbose` to see the raw LLM response
2. Check `prompts/score_candidate.txt` for ambiguous instructions
3. Check that the resume PDF is parsing correctly (`parser.py`)
4. Check that the JD extraction is picking up the right requirements (`jd_parser.py`)

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

- `src/models.py` — changing a dataclass field breaks all callers
- `src/llm_client.py` — the interface contract must stay stable
- `src/pipeline.py` — the execution order matters

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
