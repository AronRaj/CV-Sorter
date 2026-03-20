# Day 3 Goal

## What we are building today

Tests, hardening, prompt tuning, and the complete Report.docx.

Day 1 built the foundation. Day 2 built the product. Day 3 makes it
submission-ready: resilient to bad input, covered by tests, and fully
documented in the capstone report.

---

## End of day definition of done

**All of the following must be true before Day 3 is complete:**

### 1. Test suite passes
```bash
python -m pytest tests/ -v
```
Output:
```
tests/test_models.py::test_resume_dataclass          PASSED
tests/test_models.py::test_candidate_score_dataclass PASSED
tests/test_parser.py::test_pymupdf_parser_available  PASSED
tests/test_parser.py::test_parse_real_pdf            PASSED
tests/test_parser.py::test_empty_folder_returns_none PASSED
tests/test_config.py::test_load_config_defaults      PASSED
tests/test_config.py::test_missing_provider_raises   PASSED
tests/test_scorer.py::test_parse_valid_response      PASSED
tests/test_scorer.py::test_strip_code_fences         PASSED
tests/test_scorer.py::test_invalid_json_raises       PASSED
tests/test_pipeline.py::test_empty_folder_raises     PASSED

11 passed in X.XXs
```

### 2. Edge cases handled
```bash
# Empty resumes folder
python main.py --jd job_description.txt --resumes ./empty_folder --provider claude
# Expected: clear error, exit code 1, no crash

# Bad provider name
python main.py --jd job_description.txt --resumes ./resumes --provider badprovider
# Expected: clear error listing valid providers, exit code 1

# Missing JD file
python main.py --jd nonexistent.txt --resumes ./resumes --provider claude
# Expected: clear error, exit code 1
```

### 3. Report.docx fully filled
Open Report.docx — every section must have real content, not
placeholder brackets. The results section must contain an actual
comparison table from your Day 2 cross-provider runs.

---

## What gets built today (in order)

### 1. Prompt tuning pass (first, before anything else)
Review 3–5 real scoring results from Day 2. Read the LLM explanations
critically — are they specific or generic? Are the scores consistent
with what a human recruiter would say? Tighten `score_candidate.txt`
if needed. This is done before writing tests so the tests reflect the
final prompt, not a draft version.

### 2. `tests/test_models.py`
Fast unit tests — no LLM calls, no file I/O. Just construct dataclass
instances and assert their fields. Verifies `models.py` is still intact
and importable.

### 3. `tests/test_config.py`
Tests that `load_config()` correctly reads env vars, applies defaults,
and raises on missing required fields. Uses `monkeypatch` to set env
vars in test isolation — no real `.env` file needed.

### 4. `tests/test_parser.py`
Tests `PyMuPDFParser` and `ResumeParser` against real sample PDFs in
`tests/sample_resumes/`. Confirms text extraction returns non-empty
content and that `parse_all()` handles an empty folder gracefully.

### 5. `tests/test_scorer.py`
Tests `Scorer._parse_response()` and `_strip_code_fences()` in
isolation using a mocked `LLMClient`. No real API calls — the mock
returns a hardcoded valid JSON string. This makes the test suite fast
and free to run.

### 6. `tests/test_pipeline.py`
One integration-style test: confirm that `Pipeline.run()` raises
`RuntimeError` when the resumes folder is empty. Uses a real temp
folder. No LLM calls needed for this specific case.

### 7. Edge case hardening
Three specific scenarios must produce clean, helpful error messages
rather than tracebacks: empty resumes folder, unknown provider name,
and missing JD file. Review `main.py` and `pipeline.py` to confirm
all three are handled.

### 8. Report.docx — all sections
Fill every section of the provided Report.docx template using your
real project as the content. The results section uses the actual
scored JSON from your three Day 2 provider runs.

---

## Files created or modified today

```
cv_sorter/
├── tests/
│   ├── test_models.py        (NEW)
│   ├── test_config.py        (NEW)
│   ├── test_parser.py        (NEW)
│   ├── test_scorer.py        (NEW)
│   └── test_pipeline.py      (NEW)
├── tests/sample_resumes/
│   └── sample_cv.pdf         (NEW — synthetic, safe to commit)
├── prompts/
│   └── score_candidate.txt   (POSSIBLY MODIFIED — prompt tuning)
└── Report.docx               (FILLED — all sections complete)
```

Files NOT touched today: all `src/` modules are stable. If a test
reveals a bug in a Day 1 or Day 2 file, fix the bug in that file but
do not refactor its interface.

---

## Key rules for today

- Tests must not make real LLM API calls — mock the client
- Tests must not depend on `.env` being present — use `monkeypatch`
- `tests/sample_resumes/sample_cv.pdf` must be synthetic (not a real
  person's CV) — it is safe to commit to version control
- Prompt tuning edits only `prompts/score_candidate.txt` — never
  Python files
- Report.docx results section must use real numbers from your actual
  runs, not invented values

---

## If something goes wrong

| Problem | Likely cause | Fix |
|---|---|---|
| `ImportError` in tests | Test importing from wrong path | Run pytest from project root: `cd cv_sorter && python -m pytest tests/` |
| `monkeypatch` not working for env vars | `load_dotenv()` overrides monkeypatch | Pass `override=False` to `load_dotenv()` in `config.py`, or patch `os.environ` directly |
| Sample PDF extraction returns empty | PDF is image-based | Create a simple text PDF using Python's `reportlab` library instead |
| pytest not found | venv not activated | `source venv/bin/activate` |
| Report section feels thin | Not enough detail | Use the cross-provider JSON comparison — it generates rich content naturally |

---

## What Day 3 does NOT include

- No new features
- No UI additions
- No PaddleOCR implementation (that is post-MVP)
- No refactoring of Day 1 or Day 2 interfaces unless a test reveals
  an actual bug

The project is feature-complete after Day 2. Today is about quality,
confidence, and documentation.
