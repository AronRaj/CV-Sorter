# Architecture Reference

This document is the authoritative reference for the CV Sorter codebase.
Read this before making any structural changes. Designed for use with Claude Code or Cursor.

---

## Execution Flow

```
User runs: python main.py --jd job.txt --resumes ./resumes --provider claude

main.py
  └── parses CLI args
  └── loads Config
  └── calls Pipeline.run()

Pipeline.run()
  ├── 1. JDParser.parse(jd_path)          → JobDescription
  ├── 2. ResumeParser.parse_all(folder)   → List[Resume]
  ├── 3. LLMClient.build(provider)        → LLMClient (Claude|OpenAI|Gemini)
  ├── 4. Scorer.score(resume, jd, client) → CandidateScore  [for each resume]
  ├── 5. Sort CandidateScore list by overall_score DESC
  └── 6. OutputWriter.write(ranked, path) → results/ranked_output.json
```

---

## Module Contracts

Each module has a clear input/output contract. Never bypass these — always pass data through the defined interfaces.

### `src/models.py`
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

### `src/config.py`
Reads `.env` and exposes a typed `Config` object. Import this everywhere instead of calling `os.getenv` directly.

```python
@dataclass
class Config:
    default_provider: str     # "claude" | "openai" | "gemini"
    anthropic_api_key: str | None
    openai_api_key: str | None
    gemini_api_key: str | None
    output_path: str          # default: "results/ranked_output.json"
    pdf_parser: str           # "pymupdf" (default) | "paddle" (post-MVP)
    verbose: bool

def load_config() -> Config: ...
```

### `src/llm_client.py`
Abstract base + three concrete implementations. All callers use only `complete()`.

```python
class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str: ...
    
    @staticmethod
    def build(provider: str, config: Config) -> "LLMClient": ...

class ClaudeClient(LLMClient):    # anthropic SDK
class OpenAIClient(LLMClient):    # openai SDK
class GeminiClient(LLMClient):    # google-generativeai SDK
```

`LLMClient.build("claude", config)` is the factory method. Add new providers here only.

### `src/parser.py`
Abstract base class (`PDFParser`) + concrete implementations. All callers depend only on the abstract interface — never on a concrete class directly. This is the same pattern as `LLMClient`, applied to PDF parsing.

```python
class PDFParser(ABC):
    """Abstract interface all PDF parsers must implement."""

    @abstractmethod
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from a single PDF. Returns markdown string (or plain text for OCR parsers)."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this parser's dependencies are installed."""
        ...

    @staticmethod
    def build(parser_name: str) -> "PDFParser":
        """Factory method. Returns the correct implementation for the given name."""
        ...


class PyMuPDFParser(PDFParser):
    """
    Primary parser. Uses pymupdf4llm to convert digital PDFs to LLM-ready markdown.
    Handles: text, tables, multi-column layouts in text-based PDFs.
    Falls back to raw text extraction if markdown conversion fails.
    Install: pip install pymupdf4llm  (already in requirements.txt)
    """

class PaddleOCRParser(PDFParser):
    """
    Post-MVP OCR parser. Handles scanned/image PDFs that PyMuPDF cannot read.
    Uses PaddleOCR to recognise text from rendered page images.
    Install: pip install paddlepaddle paddleocr  (NOT in requirements.txt — install separately)
    Set PDF_PARSER=paddle in .env to activate.
    """
```

`ResumeParser` wraps the active `PDFParser` and adds the resume-specific logic (parse_all, logging, Resume dataclass construction). It does not know which `PDFParser` is running underneath.

```python
class ResumeParser:
    def __init__(self, parser: PDFParser): ...
    def parse(self, pdf_path: str) -> Resume: ...
    def parse_all(self, folder_path: str) -> list[Resume]: ...
```

`Pipeline` builds the correct `PDFParser` via `PDFParser.build(config.pdf_parser)` and injects it into `ResumeParser`. No other file needs to know which parser is active.

### `src/jd_parser.py`
Reads a `.txt` job description and uses the LLM to extract structured fields.

```python
class JDParser:
    def __init__(self, client: LLMClient): ...
    def parse(self, jd_path: str) -> JobDescription: ...
```

Uses `prompts/extract_jd.txt` as the prompt template.

### `src/scorer.py`
Sends each resume + JD to the LLM using `prompts/score_candidate.txt`.
Parses the JSON response into a `CandidateScore`.

```python
class Scorer:
    def __init__(self, client: LLMClient): ...
    def score(self, resume: Resume, jd: JobDescription) -> CandidateScore: ...
```

The LLM is prompted to respond in JSON. If parsing fails, retry once with a stricter prompt before raising.

### `src/pipeline.py`
Orchestrator. No business logic — only coordination.

```python
class Pipeline:
    def __init__(self, config: Config): ...
    def run(self, jd_path: str, resumes_folder: str, provider: str) -> list[CandidateScore]: ...
```

### `src/output.py`
Serialises ranked results to JSON.

```python
class OutputWriter:
    def write(self, ranked: list[CandidateScore], jd: JobDescription, provider: str, path: str) -> None: ...
```

---

## Prompt Templates

Templates live in `prompts/` as plain text files. Python uses `.format(**vars)` to fill placeholders.

### `prompts/score_candidate.txt`
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

### `prompts/extract_jd.txt`
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

### Why a single `complete()` interface?
All three LLM providers (Claude, OpenAI, Gemini) accept a prompt string and return a string. Keeping the interface minimal means the scorer never needs to know which provider it is using. Swapping providers requires no changes outside `llm_client.py`.

### Why does `PDFParser` use the same abstract interface pattern as `LLMClient`?
Both solve the same problem: multiple implementations of the same operation that need to be swappable at config time with zero changes to callers. `ResumeParser` calls `parser.extract_text(path)` — it does not know or care whether it is running `PyMuPDF` or `PaddleOCR`. This means adding PaddleOCR post-MVP is a two-file change: add the implementation in `parser.py`, add the factory case in `PDFParser.build()`. Nothing else changes.

### Why pymupdf4llm as the primary parser (not pdfplumber or PaddleOCR)?
`pymupdf4llm` outputs markdown — tables become markdown tables, columns are correctly ordered, headings are preserved. Sending structured markdown to the LLM gives noticeably better scoring quality than sending a wall of jumbled plain text. `pdfplumber` outputs plain text only and struggles with multi-column CV layouts. PaddleOCR is powerful but requires a 2GB deep learning framework and is slow without a GPU — that cost is not justified for digital PDFs, which represent the majority of modern CVs.

### Why no automatic scanned-PDF fallback in the MVP?
A silent fallback (try pymupdf → if empty, try PaddleOCR) would require PaddleOCR to always be installed, which reintroduces the heavy dependency for everyone. Instead, the MVP logs a clear warning when pymupdf returns empty text, tells the user which file failed, and suggests enabling the paddle parser. This is the right tradeoff for a capstone project.

### Why dataclasses instead of dicts?
Mirrors Kotlin data classes. Gives autocompletion, type checking, and clear contracts between modules. All dataclasses are in `models.py` — one file to look at when debugging data shape issues.

### Why are prompts in `.txt` files?
Prompt engineering is iterative. Keeping templates outside Python means you can improve scoring quality without touching the codebase. Claude Code / Cursor can also be pointed directly at a prompt file to improve it.

### Why JSON-only LLM responses?
Structured output is easier to parse reliably than prose. The scorer instructs the LLM to return only valid JSON and implements one retry on parse failure. This is the most fragile part of the system — if scoring breaks, check the prompt template first.

---

## Adding a New Provider

1. Add the API key to `.env.example` and `Config` in `config.py`
2. Create a new class in `llm_client.py` that extends `LLMClient` and implements `complete()`
3. Add a case in `LLMClient.build()` for the new provider name
4. Add the provider name to the `--provider` choices in `main.py`
5. Update the providers table in `README.md`

No other files need to change.

---

## Plugging in PaddleOCR (Post-MVP)

This is a two-file change. Nothing else in the codebase needs to be touched.

**Step 1 — Install PaddleOCR separately** (not in requirements.txt — too heavy for default install):
```bash
pip install paddlepaddle paddleocr
```

**Step 2 — Implement `PaddleOCRParser` in `src/parser.py`**:
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
        # Convert each PDF page to an image, run OCR, join results
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
    elif parser_name == "paddle":          # ← add this case
        return PaddleOCRParser()
    else:
        raise ValueError(f"Unknown parser: {parser_name}")
```

**Step 4 — Set in `.env`**:
```
PDF_PARSER=paddle
```

That's it. `ResumeParser`, `Pipeline`, and every other module are completely unaffected.

---

## Error Handling Strategy

| Error type              | Where handled     | Behaviour                                        |
|-------------------------|-------------------|--------------------------------------------------|
| Missing API key         | `config.py`       | Raise at startup with a clear message            |
| PDF parse failure       | `parser.py`       | Log warning, skip file, continue with others     |
| Scanned PDF detected    | `parser.py`       | Log warning with filename, suggest `PDF_PARSER=paddle` in `.env`, skip file |
| LLM JSON parse failure  | `scorer.py`       | Retry once with stricter prompt, then raise      |
| All resumes fail        | `pipeline.py`     | Raise with summary of failures                   |
| Output folder missing   | `output.py`       | Create folder, then write                        |

---

## What Not to Do

- Do not call `os.getenv()` outside `config.py`
- Do not put prompt text inside Python strings — use `prompts/*.txt`
- Do not add UI/web code to any file in `src/` — the CLI layer lives in `main.py` only
- Do not import from `pipeline.py` into individual modules — dependency flows one way only
- Do not store API keys or resume data in version control
