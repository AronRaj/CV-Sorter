# Day 1 Goal

## What we are building today

The data foundation and parsing layer of the CV Sorter.

No scoring. No CLI. No full pipeline. Today is about making sure the
building blocks are correct before any LLM calls that cost money.

---

## End of day definition of done

Originally this was checked with a temporary `verify_day1.py` script (removed
after Day 1 — use `python -m pytest tests/` and the CLI in `main.py` now).
The expected output when exercising Config, JD parsing, resumes, and an LLM
smoke test looked like this:

```
=== Config loaded ===
Provider : claude
Parser   : pymupdf
Output   : results/ranked_output.json

=== Job Description parsed ===
Title       : Senior Backend Engineer
Required    : ['Python', 'REST APIs', 'PostgreSQL']
Nice to have: ['Docker', 'Kubernetes']
Min exp     : 3 years

=== Resumes parsed: 2 found ===
[1] john_doe.pdf     — 842 words extracted
[2] jane_smith.pdf   — 1103 words extracted

=== LLM smoke test ===
Claude responded: Hello! I am ready to score resumes.

Day 1 complete.
```

If you see all four sections printed without errors, Day 1 is done.

---

## What gets built today (in order)

### 1. Project scaffold
Folder structure, virtual environment, dependencies installed, `.env` file
in place with at least one API key. Nothing runs yet — just the skeleton.

### 2. `src/models.py`
All dataclasses for the project. No logic, no imports from other src files.
This is the single source of truth for every data shape used across the system.
Think of it as the POJO layer in a Java/Kotlin project.

### 3. `src/config.py`
Reads `.env` and returns a typed `Config` dataclass. Every other module
imports `Config` from here — no module ever calls `os.getenv()` directly.
Raises a clear error at startup if a required key is missing.

### 4. `src/parser.py`
Two things in one file:
- `PDFParser` — abstract base class (like a Kotlin interface)
- `PyMuPDFParser` — the concrete MVP implementation using `pymupdf4llm`
- `PaddleOCRParser` — stub only today (raises NotImplementedError)
- `ResumeParser` — wraps whichever `PDFParser` is active, adds parse_all()

### 5. `src/llm_client.py`
- `LLMClient` — abstract base class with one method: `complete(prompt) -> str`
- `ClaudeClient` — concrete implementation using the `anthropic` SDK
- `OpenAIClient` — stub only today
- `GeminiClient` — stub only today
- `LLMClient.build()` — factory method

Only Claude is fully wired today. OpenAI and Gemini are Day 2.

### 6. `src/jd_parser.py`
Reads a `.txt` job description file, fills the `prompts/extract_jd.txt`
template, calls the LLM, parses the JSON response into a `JobDescription`
dataclass. This is the first real LLM call in the project.

### 7. Sample data
- `job_description.txt` — a realistic software engineering JD
- Two sample resume PDFs in `resumes/` (can be your own CV or downloaded samples)

### 8. Day 1 verification (historical)
A temporary `verify_day1.py` script exercised the modules above in sequence.
It has been removed from the repo; integration is covered by `tests/` and the
full pipeline CLI instead.

---

## Files created today

```
cv_sorter/
├── src/
│   ├── __init__.py       (empty)
│   ├── models.py         (NEW — all dataclasses)
│   ├── config.py         (NEW — Config loader)
│   ├── parser.py         (NEW — PDFParser + PyMuPDFParser + ResumeParser)
│   ├── llm_client.py     (NEW — LLMClient ABC + ClaudeClient)
│   └── jd_parser.py      (NEW — JDParser)
├── resumes/
│   ├── john_doe.pdf      (NEW — sample resume)
│   └── jane_smith.pdf    (NEW — sample resume)
├── job_description.txt   (NEW — sample JD)
└── .env                  (NEW — from .env.example, API key added)
```

Files NOT touched today: `main.py`, `pipeline.py`, `scorer.py`, `output.py`

---

## Key rules for today

- `src/models.py` must not import from any other `src/` file
- No `os.getenv()` calls outside `config.py`
- `ResumeParser` must not know whether it is using PyMuPDF or PaddleOCR —
  it only calls `self._parser.extract_text(path)`
- `ClaudeClient.complete()` must call the real Anthropic API —
  no mocking today, we need to confirm the key works
- Type hints on every function signature
- Docstrings on every class and public method

---

## If something goes wrong

| Problem | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: pymupdf4llm` | venv not activated or pip install missed | `source venv/bin/activate && pip install -r requirements.txt` |
| `AuthenticationError` from Anthropic | Wrong or missing API key | Check `.env` — key must start with `sk-ant-` |
| PDF parses to empty string | Scanned/image PDF | Use a different PDF, or a text-based one created in Word/Google Docs |
| `KeyError` in config | Field missing from `.env` | Compare your `.env` against `.env.example` |

---

## What Day 1 does NOT include

- No CLI (`main.py` is not written yet)
- No scoring (`scorer.py` is not written yet)
- No JSON output (`output.py` is not written yet)
- No OpenAI or Gemini (stubs only — fully wired on Day 2)
- No pytest tests (written on Day 3)

Those are deliberately deferred. The goal today is a rock-solid foundation
that you have personally verified works end to end.