# Architecture Reference

This document is the authoritative reference for the CV Sorter codebase.
Read this before making any structural changes. Designed for use with Claude Code or Cursor.

---

## Execution Flow

```
User runs: python main.py

main.py
  └── parses optional CLI args (all have defaults)
  └── loads Config from .env
  └── calls Supervisor(config=config)
        └── builds models internally:
              get_claude_model(config)       → scorer + report
              get_ollama_shortlist_model()   → shortlist agent
              get_ollama_jd_model()          → JD extraction

Supervisor.run()
  ├── 1. ResumeParser.parse_all(folder)           → list[Resume]
  ├── 2. JDParser(jd_model).parse(jd_path)        → JobDescription
  ├── 3. ShortlistAgent.run(resumes, jd)           → shortlisted, skipped
  ├── 4. ScorerAgent.run(shortlisted, jd)          → list[CandidateScore]  (sorted)
  ├── 5. ReportAgent.run(scored, jd)               → recruiter_summary.md
  └── 6. OutputWriter.write(scored, jd, path)      → results/ranked_output.json
```

---

## Two-Layer Architecture

```
src/core/   — domain layer
  Pure business logic. No agents, no LangChain, no orchestration.
  Called by src/agents/ through tools.py. Tested directly in tests/.

  models.py              dataclasses — shared types for the entire system
  config.py              env var loading — typed Config dataclass
  parser_engine.py       PDF → Resume (PDFParser ABC + PyMuPDF + PaddleOCR stub)
  scorer_engine.py       resume + JD → CandidateScore (accepts BaseChatModel)
  jd_parser_engine.py    JD text → JobDescription (accepts BaseChatModel)
  output_engine.py       CandidateScore list → JSON + CSV + Rich table

src/agents/  — application layer
  Orchestration and agent behaviour. Imports from src/core/.
  Never imported by src/core/.

  model_factory.py   named model constructors (Claude + Ollama only)
  tools.py           @tool wrappers around src/core/ modules
  shortlist_agent.py fast-pass filter using local model
  scorer_agent.py    deep scoring with self-evaluation loop
  report_agent.py    cross-candidate recruiter synthesis
  supervisor.py      sequential coordinator: parse → shortlist → score → report

src/prompts/  — prompt templates (.txt, filled with .format())
src/results/  — output directory (git-ignored contents)
src/resumes/  — input PDF resumes
```

---

## Module Contracts

Each module has a clear input/output contract. Never bypass these — always pass data through the defined interfaces.

### `src/core/models.py`
Single source of truth for all data shapes. No logic — only dataclasses.

```python
@dataclass
class Resume:
    filename: str
    raw_text: str
    name: str | None
    skills: list[str]
    experience_years: int | None
    education: list[str]

@dataclass
class JobDescription:
    raw_text: str
    job_title: str
    required_skills: list[str]
    nice_to_have_skills: list[str]
    min_experience_years: int | None
    responsibilities: list[str]

@dataclass
class RequirementScore:
    requirement: str
    score: int           # 0–100
    evidence: str        # LLM explanation

@dataclass
class CandidateScore:
    resume: Resume
    overall_score: int   # 0–100
    fit_label: str       # "Strong match" | "Good match" | "Partial match" | "Weak match"
    explanation: str
    requirement_scores: list[RequirementScore]
```

### `src/core/config.py`
Reads `.env` and exposes a typed `Config` object. Import this everywhere instead of calling `os.getenv` directly.

```python
OLLAMA_SHORTLIST_MODEL: str = "llama3.1:latest"
OLLAMA_JD_MODEL: str = "gemma2:latest"

@dataclass
class Config:
    pdf_parser: str           # "pymupdf" (default) | "paddle" (post-MVP)
    output_path: str          # default: "results/ranked_output.json"
    verbose: bool
    anthropic_api_key: str | None
    langchain_tracing_v2: str # default: "false"
    langchain_api_key: str | None
    langchain_project: str    # default: "cv-sorter-agents"

def load_config() -> Config: ...
```

### `src/core/parser_engine.py`
Abstract base class (`PDFParser`) + concrete implementations. All callers depend only on the abstract interface — never on a concrete class directly.

```python
class PDFParser(ABC):
    @abstractmethod
    def extract_text(self, pdf_path: str) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @staticmethod
    def build(parser_name: str) -> "PDFParser": ...

class PyMuPDFParser(PDFParser):
    """Primary parser. Uses pymupdf4llm to convert digital PDFs to markdown."""

class PaddleOCRParser(PDFParser):
    """Post-MVP OCR parser for scanned/image PDFs."""
```

`ResumeParser` wraps the active `PDFParser` and adds the resume-specific logic (parse_all, logging, Resume dataclass construction). It does not know which `PDFParser` is running underneath.

```python
class ResumeParser:
    def __init__(self, parser: PDFParser): ...
    def parse(self, pdf_path: str) -> Resume: ...
    def parse_all(self, folder_path: str) -> list[Resume]: ...
```

### `src/core/jd_parser_engine.py`
Reads a `.txt` job description and uses the LLM to extract structured fields.

```python
class JDParser:
    def __init__(self, model: BaseChatModel): ...
    def parse(self, jd_path: str) -> JobDescription: ...
```

Uses `src/prompts/extract_jd.txt` as the prompt template.

### `src/core/scorer_engine.py`
Sends each resume + JD to the LLM using `src/prompts/score_candidate.txt`.
Parses the JSON response into a `CandidateScore`.

```python
class Scorer:
    def __init__(self, model: BaseChatModel): ...
    def score(self, resume: Resume, jd: JobDescription) -> CandidateScore: ...
```

The LLM is prompted to respond in JSON. If parsing fails, retry once with a stricter prompt before raising.

### `src/core/output_engine.py`
Serialises ranked results to JSON, CSV, and Rich terminal table.

```python
class OutputWriter:
    def write(self, ranked: list[CandidateScore], jd: JobDescription,
              provider: str, model: str, output_path: str) -> None: ...
    def write_csv(self, ranked: list[CandidateScore], output_path: str) -> None: ...
    def print_table(self, ranked: list[CandidateScore]) -> None: ...
```

### `src/agents/model_factory.py`
Named model constructors for the two supported providers: Claude (API) and Ollama (local).
Model assignments are fixed architectural decisions, not runtime options.

```python
CLAUDE_MODEL: str = "claude-sonnet-4-5"
TEMPERATURE: float = 0.0

def get_claude_model(config: Config) -> BaseChatModel: ...
def get_ollama_shortlist_model() -> BaseChatModel: ...
def get_ollama_jd_model() -> BaseChatModel: ...
```

### `src/agents/supervisor.py`
Sequential coordinator. Builds all models internally using the named factory functions.
The caller passes only a Config — model assignments are fixed architectural decisions.

```python
class Supervisor:
    def __init__(self, config: Config) -> None: ...
    def run(self, jd_path: str, resumes_folder: str,
            output_json_path: str, output_summary_path: str) -> dict: ...
```

### `src/agents/shortlist_agent.py`
Fast first-pass filter. Uses a local model to screen resumes against the JD
and returns two lists: shortlisted (proceed to scoring) and skipped.

```python
class ShortlistAgent:
    def __init__(self, model: BaseChatModel) -> None: ...
    def run(self, resumes: list[Resume], jd: JobDescription) -> tuple[list[Resume], list[Resume]]: ...
```

### `src/agents/scorer_agent.py`
Deep scoring agent. Scores each resume against the JD using `src/core/scorer_engine.py`,
then runs a self-evaluation loop to flag low-confidence scores for re-scoring.

```python
class ScorerAgent:
    def __init__(self, model: BaseChatModel) -> None: ...
    def run(self, resumes: list[Resume], jd: JobDescription) -> list[CandidateScore]: ...
```

### `src/agents/report_agent.py`
Synthesises a recruiter-facing markdown summary across all scored candidates.

```python
class ReportAgent:
    def __init__(self, model: BaseChatModel) -> None: ...
    def run(self, results: list[CandidateScore], jd: JobDescription,
            output_path: str) -> str: ...
```

---

## Prompt Templates

Templates live in `src/prompts/` as plain text files. Python uses `.format(**vars)` to fill placeholders.

### `src/prompts/score_candidate.txt`
Placeholders: `{job_title}`, `{required_skills}`, `{nice_to_have_skills}`, `{responsibilities}`, `{resume_text}`

The LLM must respond with **only valid JSON**, no preamble. The scorer parses this directly.

Expected response shape:
```json
{
  "overall_score": 82,
  "fit_label": "Strong match",
  "explanation": "...",
  "requirement_scores": [
    { "requirement": "Python 5+ years", "score": 90, "evidence": "..." }
  ]
}
```

### `src/prompts/extract_jd.txt`
Placeholders: `{raw_jd_text}`

Expected response shape:
```json
{
  "job_title": "...",
  "required_skills": ["...", "..."],
  "nice_to_have_skills": ["..."],
  "min_experience_years": 3,
  "responsibilities": ["...", "..."]
}
```

---

## Key Design Decisions

### Why remove LLMClient ABC in favour of LangChain BaseChatModel?
`BaseChatModel` supports both providers (Claude and Ollama) through one interface, with native tool support, structured output parsing, and LangSmith tracing built in. The hand-rolled `LLMClient` ABC added complexity — separate SDK wrappers for each provider, no tracing, no tool support — with no benefit once LangChain was adopted for the agent layer. Removing it eliminated a redundant abstraction and unified the entire codebase on one LLM interface.

### Why only two providers (Claude + Ollama)?
Local models (Ollama) handle tasks where speed and zero cost matter more than reasoning quality — shortlisting and JD extraction. Claude handles tasks where output quality is directly visible to the recruiter — scoring and the recruiter briefing. Adding more API providers (OpenAI, Gemini, etc.) would add configuration complexity with no functional benefit for the pipeline. The model assignments are fixed architectural decisions, not runtime options.

### Why does `PDFParser` use an abstract interface pattern?
Multiple implementations of the same operation need to be swappable at config time with zero changes to callers. `ResumeParser` calls `parser.extract_text(path)` — it does not know or care whether it is running `PyMuPDF` or `PaddleOCR`. This means adding PaddleOCR post-MVP is a two-file change: add the implementation in `parser.py`, add the factory case in `PDFParser.build()`. Nothing else changes.

### Why pymupdf4llm as the primary parser (not pdfplumber or PaddleOCR)?
`pymupdf4llm` outputs markdown — tables become markdown tables, columns are correctly ordered, headings are preserved. Sending structured markdown to the LLM gives noticeably better scoring quality than sending a wall of jumbled plain text. `pdfplumber` outputs plain text only and struggles with multi-column CV layouts. PaddleOCR is powerful but requires a 2GB deep learning framework and is slow without a GPU — that cost is not justified for digital PDFs, which represent the majority of modern CVs.

### Why no automatic scanned-PDF fallback in the MVP?
A silent fallback (try pymupdf → if empty, try PaddleOCR) would require PaddleOCR to always be installed, which reintroduces the heavy dependency for everyone. Instead, the MVP logs a clear warning when pymupdf returns empty text, tells the user which file failed, and suggests enabling the paddle parser. This is the right tradeoff for a capstone project.

### Why a multi-agent architecture instead of a single pipeline?
The single-pipeline approach scored every resume with a full API call. The multi-agent design uses a cheap local model (Ollama) to screen out clearly unqualified candidates first, then only sends shortlisted resumes to the expensive API model. This reduces API costs, improves latency, and enables a self-evaluation loop where the scorer agent can flag and re-score low-confidence results.

### Why dataclasses instead of dicts?
Mirrors Kotlin data classes. Gives autocompletion, type checking, and clear contracts between modules. All dataclasses are in `models.py` — one file to look at when debugging data shape issues.

### Why are prompts in `.txt` files?
Prompt engineering is iterative. Keeping templates outside Python means you can improve scoring quality without touching the codebase. Claude Code / Cursor can also be pointed directly at a prompt file to improve it.

### Why JSON-only LLM responses?
Structured output is easier to parse reliably than prose. The scorer instructs the LLM to return only valid JSON and implements one retry on parse failure. This is the most fragile part of the system — if scoring breaks, check the prompt template first.

---

## Adding a New Provider

1. Add the API key env var to `.env.example` and the `Config` dataclass in `src/core/config.py`
2. Add a named factory function in `src/agents/model_factory.py` (e.g. `get_groq_model(config)`) that returns a `BaseChatModel`
3. Update `Supervisor.__init__` in `src/agents/supervisor.py` to call the new factory function for the appropriate task
4. Update `README.md` model choices table

No other files need to change.

---

## Plugging in PaddleOCR (Post-MVP)

This is a two-file change. Nothing else in the codebase needs to be touched.

**Step 1 — Install PaddleOCR separately** (not in requirements.txt — too heavy for default install):
```bash
pip install paddlepaddle paddleocr
```

**Step 2 — Implement `PaddleOCRParser` in `src/core/parser_engine.py`**:
```python
class PaddleOCRParser(PDFParser):
    def is_available(self) -> bool:
        try:
            import paddleocr
            return True
        except ImportError:
            return False

    def extract_text(self, pdf_path: str) -> str:
        from paddleocr import PaddleOCR
        import pdf2image
        pages = pdf2image.convert_from_path(pdf_path)
        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        all_text: list[str] = []
        for page_image in pages:
            result = ocr.ocr(page_image, cls=True)
            page_lines = [line[1][0] for block in result for line in block]
            all_text.extend(page_lines)
        return "\n".join(all_text)
```

**Step 3 — Add the factory case in `PDFParser.build()`**:
```python
@staticmethod
def build(parser_name: str) -> "PDFParser":
    if parser_name == "pymupdf":
        return PyMuPDFParser()
    elif parser_name == "paddle":
        return PaddleOCRParser()
    else:
        raise ValueError(f"Unknown parser: {parser_name}")
```

**Step 4 — Set in `.env`**:
```
PDF_PARSER=paddle
```

That's it. `ResumeParser`, `Supervisor`, and every other module are completely unaffected.

---

## Error Handling Strategy

| Error type              | Where handled       | Behaviour                                        |
|-------------------------|---------------------|--------------------------------------------------|
| Missing API key         | `src/agents/model_factory.py`  | Raise at startup with a clear message            |
| PDF parse failure       | `src/core/parser_engine.py`    | Log warning, skip file, continue with others     |
| Scanned PDF detected    | `src/core/parser_engine.py`    | Log warning with filename, suggest `PDF_PARSER=paddle` in `.env`, skip file |
| LLM JSON parse failure  | `src/core/scorer_engine.py`    | Retry once with stricter prompt, then raise      |
| All resumes fail        | `src/agents/supervisor.py`     | Raise with summary of failures                   |
| Output folder missing   | `src/core/output_engine.py`    | Create folder, then write                        |

---

## What Not to Do

- Do not call `os.getenv()` outside `src/core/config.py`
- Do not put prompt text inside Python strings — use `src/prompts/*.txt`
- Do not add UI/web code to any file in `src/` — the CLI layer lives in `main.py` only
- Do not import from `src/agents/` into `src/core/` modules — dependency flows one way: `src/agents/` → `src/core/`, never reverse
- Do not store API keys or resume data in version control
