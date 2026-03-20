# Day 4 Goal

## What we are building today

The presentation and interactivity layer that satisfies the capstone
requirement section 4: "Intelligent Ranking and User Interface."

Three things get built:
1. CSV export — a recruiter-friendly exportable file format
2. Streamlit dashboard — ranked table with evidence drill-down
3. Interactive re-ranking — skill weight sliders that re-sort
   candidates in real time from cached JSON, with zero LLM calls

The core pipeline (Days 1–3) is NOT touched. Everything today plugs
in at the edges: one new method in `output.py`, three new CLI flags
in `main.py`, and one new top-level file `streamlit_app.py`.

---

## End of day definition of done

### Terminal — CSV export works
```bash
python main.py \
  --jd job_description.txt \
  --resumes ./resumes \
  --provider claude \
  --export-csv results/ranked.csv
```
Expected: `results/ranked.csv` created. Open it in Excel or Numbers —
one row per candidate, columns: Rank, Filename, Score, Fit, Explanation.

### Terminal — filter flags work
```bash
# Only show candidates scoring 70 or above
python main.py --jd job_description.txt --resumes ./resumes \
  --provider claude --verbose --min-score 70

# Score all, but only show candidates who scored >= 60 on Python
python main.py --jd job_description.txt --resumes ./resumes \
  --provider claude --verbose --filter-skills "Python"
```
Expected: terminal table shows only matching candidates. JSON still
contains all candidates unfiltered.

### Streamlit — dashboard launches
```bash
streamlit run streamlit_app.py
```
Expected: browser opens at `http://localhost:8501` and shows:

```
CV Sorter — Recruiter Dashboard

[Load results file]  results/ranked_claude.json   [Load]

Job: Senior Backend Engineer  |  Provider: claude  |  3 candidates

──────────────────────────────────────────────────────
#1  john_doe.pdf          87/100   ● Strong match
    ▶ Show evidence                               [▼]
      Python 5+ years      90/100  "6 years Python stated in summary"
      PostgreSQL            75/100  "Listed under core skills"
      ...

#2  jane_smith.pdf        71/100   ● Good match
    ▶ Show evidence                               [▼]
──────────────────────────────────────────────────────

─── Interactive Re-ranking ───────────────────────────
Skill weights (drag to adjust importance):
  Python 5+ years     [━━━━━━━━━━━━━━━] 1.0×
  PostgreSQL          [━━━━━━━━━━━━━━━] 1.0×
  Docker              [━━━━━━━━━━━━━━━] 1.0×

[Re-rank]  →  table updates instantly, no LLM call
──────────────────────────────────────────────────────
```

All three sections must render correctly before Day 4 is done.

---

## What gets built today (in order)

### 1. Install Streamlit
Add `streamlit` to `requirements.txt` and install it.

### 2. `write_csv()` in `src/output.py`
One new method alongside the existing `write()` and `print_table()`.
Writes a flat CSV file — one row per candidate, key columns only.
No changes to existing methods.

### 3. CLI flags in `main.py`
Three new optional flags: `--export-csv`, `--min-score`, `--filter-skills`.
They are applied after `pipeline.run()` returns — the pipeline itself
is not changed. The JSON output always contains all candidates
unfiltered. CSV and terminal table respect the filters.

### 4. `streamlit_app.py` — Part A: Load and display
File picker for `ranked_output.json`, ranked candidate table with
score badges, expandable evidence rows per requirement. Read-only
view of the existing JSON data.

### 5. `streamlit_app.py` — Part B: Interactive re-ranking
Sliders (0.0×–2.0×) for each requirement extracted from the loaded
JSON. A "Re-rank" button recomputes weighted scores client-side and
re-sorts the table without making any LLM calls.

### 6. `streamlit_app.py` — Part C: Min-score filter
A slider in the sidebar to hide candidates below a score threshold.
Works together with the weight sliders.

---

## The re-ranking algorithm (important to understand)

The Streamlit app re-ranks using only data already in the JSON.
No API calls, no re-scoring.

```
For each candidate:
  weighted_score = 0
  total_weight   = 0

  For each requirement_score in candidate.requirement_scores:
      weight        = slider_value[requirement]   # default 1.0
      weighted_score += requirement_score.score * weight
      total_weight  += weight

  normalised_score = round(weighted_score / total_weight)
                     if total_weight > 0 else 0
  fit_label        = derive from normalised_score using standard bands
```

Fit label bands (same as scorer):
- 80–100 → Strong match
- 60–79  → Good match
- 40–59  → Partial match
- 0–39   → Weak match

This means:
- Setting Python weight to 2.0× doubles its contribution to the score
- Setting Docker weight to 0.0× removes it from consideration entirely
- The re-ranked order can differ from the original LLM-scored order
- This is a feature, not a bug — it lets recruiters prioritise their
  most important requirements

---

## Files created or modified today

```
cv_sorter/
├── streamlit_app.py         (NEW — the entire Streamlit dashboard)
├── main.py                  (MODIFIED — 3 new flags)
├── requirements.txt         (MODIFIED — add streamlit)
└── src/
    └── output.py            (MODIFIED — add write_csv() method)
```

Files NOT touched today:
  models.py, config.py, parser.py, llm_client.py,
  jd_parser.py, scorer.py, pipeline.py, all tests, all prompts.

The Day 1–3 test suite must still pass after Day 4 changes:
```bash
python -m pytest tests/ -v
```

---

## Key rules for today

- `streamlit_app.py` reads JSON directly — it does NOT import
  from `src/pipeline.py` or call the LLM
- `streamlit_app.py` MAY import from `src/models.py` if needed,
  but it is fine to work with plain dicts from `json.load()` too
- Re-ranking logic lives entirely in `streamlit_app.py` —
  it does not belong in `src/output.py` or `src/pipeline.py`
- The JSON output from `pipeline.run()` is always the full unfiltered
  list — filters only affect what is displayed or exported
- `--min-score` and `--filter-skills` filter the terminal table and
  the CSV export, but never the JSON file
- All existing tests must still pass after today's changes

---

## If something goes wrong

| Problem | Likely cause | Fix |
|---|---|---|
| `streamlit: command not found` | Not installed or venv not active | `source venv/bin/activate && pip install streamlit` |
| Streamlit shows blank page | JSON file not loaded yet | Click the Load button after entering the file path |
| Re-rank gives same order | All weights still at 1.0 | Drag a slider to a different value before clicking Re-rank |
| CSV opens garbled in Excel | Encoding issue | Use `encoding="utf-8-sig"` in `csv.writer` — the BOM makes Excel happy |
| `--filter-skills` matches nothing | Case mismatch | Normalise both sides to lowercase before comparing |
| Streamlit slider jumps to 0 | Step size too small | Use `step=0.1` on `st.slider()` |
| Port 8501 already in use | Another Streamlit instance running | `pkill -f streamlit` then re-run |

---

## What Day 4 does NOT include

- No database — results still live in JSON files
- No user authentication — dashboard is local-only
- No deployment (Heroku, cloud, etc.)
- No re-scoring via LLM from the dashboard — re-ranking only
- No changes to the scoring logic or prompt templates
- No new pytest tests for Streamlit (Streamlit UI testing requires
  additional tooling beyond this project scope — mention this
  limitation in the report)
