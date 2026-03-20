# Day 2 Goal

## What we are building today

The scoring engine, the pipeline that connects everything, the CLI entry
point, and all three LLM providers fully wired.

Day 1 gave you correct building blocks that you verified individually.
Day 2 connects them into a single command that produces a real ranked
JSON file from real resumes.

---

## End of day definition of done

Run this command:

```bash
python main.py --jd job_description.txt --resumes ./resumes --provider claude --verbose
```

And see output like this in your terminal:

```
CV Sorter — provider: claude | model: claude-sonnet-4-5
Parsing job description...  done
Parsing resumes...          2 found
Scoring candidates...
  [1/2] john_doe.pdf        scored
  [2/2] jane_smith.pdf      scored
Ranking...                  done
Results written to results/ranked_output.json

┌──────┬──────────────────┬───────┬────────────────┐
│ Rank │ Candidate        │ Score │ Fit            │
├──────┼──────────────────┼───────┼────────────────┤
│  1   │ john_doe.pdf     │  87   │ Strong match   │
│  2   │ jane_smith.pdf   │  61   │ Good match     │
└──────┴──────────────────┴───────┴────────────────┘
```

Then run it again twice more:

```bash
python main.py --jd job_description.txt --resumes ./resumes --provider openai --verbose
python main.py --jd job_description.txt --resumes ./resumes --provider gemini --verbose
```

All three must produce a valid `results/ranked_output.json`. If you see
ranked output from all three providers, Day 2 is done.

---

## What gets built today (in order)

### 1. `src/scorer.py`
The heart of the project. Takes one `Resume` + one `JobDescription`,
fills `prompts/score_candidate.txt`, sends it to the LLM, and parses
the JSON back into a `CandidateScore`. Includes one automatic retry
with a stricter prompt if JSON parsing fails the first time.

### 2. `src/output.py`
Takes the ranked list of `CandidateScore` objects and writes a clean
`results/ranked_output.json`. Creates the `results/` folder if it does
not exist. Also responsible for the `--verbose` rich terminal table.

### 3. `src/pipeline.py`
The orchestrator. Calls every Day 1 module in the correct order, passes
outputs as inputs to the next step. No logic of its own — only coordination.

### 4. `main.py`
The CLI entry point. Parses `--jd`, `--resumes`, `--provider`, `--output`,
`--verbose` flags using `argparse`. Calls `Pipeline.run()`. Prints a
clean progress line for each step.

### 5. OpenAIClient + GeminiClient (completing `src/llm_client.py`)
Replace the Day 1 stubs with real implementations. Add the API keys to
`.env`. Verify each provider independently before running the full pipeline.

### 6. Cross-provider comparison
Run all three providers on the same resumes. Open the three output JSON
files side by side. Note the score differences — this becomes the most
interesting section of your Report.docx results.

---

## Files created or modified today

```
cv_sorter/
├── src/
│   ├── scorer.py         (NEW)
│   ├── output.py         (NEW)
│   ├── pipeline.py       (NEW)
│   └── llm_client.py     (MODIFIED — OpenAI + Gemini stubs → real impls)
├── main.py               (NEW)
└── .env                  (MODIFIED — add OPENAI_API_KEY + GEMINI_API_KEY)
```

Files NOT touched today: `models.py`, `config.py`, `parser.py`,
`jd_parser.py`, all prompt templates. Day 1 output is treated as stable.

---

## Key rules for today

- `pipeline.py` contains zero business logic — it only calls other modules
- `scorer.py` must retry exactly once on JSON parse failure before raising
- `output.py` is the only file that writes to disk (besides logging)
- `main.py` is the only file that calls `sys.exit()`
- The `--verbose` rich table prints to terminal only — it does not go in the JSON
- All three provider clients use the exact same `complete(prompt) -> str` interface
- Never call `os.getenv()` in any new file — use `Config` from `config.py`

---

## Provider model strings to use

| Provider | Model string               |
|----------|---------------------------|
| Claude   | `claude-sonnet-4-5`        |
| OpenAI   | `gpt-4o`                   |
| Gemini   | `gemini-1.5-pro`           |

These are already decided — do not use other model strings.

---

## If something goes wrong

| Problem | Likely cause | Fix |
|---|---|---|
| `JSONDecodeError` in scorer | LLM added markdown fences or preamble | `_strip_code_fences()` helper — same pattern as `jd_parser.py` |
| Score always 0 or null | Wrong JSON field name in parser | Print raw LLM response, check field names match `CandidateScore` |
| `results/` not created | `output.py` missing `mkdir` | Use `Path(output_path).parent.mkdir(parents=True, exist_ok=True)` |
| Gemini auth error | API key format issue | Gemini keys start with `AIza` — confirm in Google AI Studio |
| OpenAI `RateLimitError` | Free tier quota | Wait 60s and retry, or use a paid key |
| Rich table garbled | Terminal does not support unicode | Pass `--no-verbose` or set `Console(force_terminal=True)` |

---

## What Day 2 does NOT include

- No pytest tests (written on Day 3)
- No prompt tuning pass (done at the start of Day 3)
- No Report.docx writing
- No `verify_day2.py` script — `main.py --verbose` is the verification

Day 3 starts from a fully working CLI that all three providers can run.
