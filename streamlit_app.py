"""CV Sorter — Recruiter Dashboard.

Streamlit app that loads ranked JSON output from the CV Sorter pipeline
and displays an interactive recruiter-facing dashboard. Reads cached
JSON only — no LLM calls, no pipeline imports.

Launch with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json

import streamlit as st


def score_badge(score: int) -> str:
    """Return an HTML badge span colour-coded by score band.

    80-100 green, 60-79 blue, 40-59 orange, 0-39 red.
    """
    if score >= 80:
        colour = "#1D9E75"
    elif score >= 60:
        colour = "#185FA5"
    elif score >= 40:
        colour = "#BA7517"
    else:
        colour = "#A32D2D"
    return (
        f'<span style="background:{colour};color:white;'
        f'padding:2px 10px;border-radius:12px;font-size:13px;">'
        f"{score}</span>"
    )


def compute_weighted_score(
    candidate: dict,
    weights: dict[str, float],
) -> int:
    """Re-compute a candidate's overall score using requirement weights.

    For each requirement, multiplies the raw score by the corresponding
    weight and normalises by the sum of weights.

    Args:
        candidate: A candidate dict from the loaded JSON.
        weights: Mapping of requirement name to weight multiplier.

    Returns:
        Weighted score as an integer 0-100.
    """
    total_weighted = 0.0
    total_weight = 0.0
    for req in candidate["requirement_scores"]:
        weight = weights.get(req["requirement"], 1.0)
        total_weighted += req["score"] * weight
        total_weight += weight
    if total_weight == 0:
        return 0
    return round(total_weighted / total_weight)


def derive_fit_label(score: int) -> str:
    """Derive fit label string from a 0-100 score.

    Uses the same bands as the scorer:
    80-100 Strong match, 60-79 Good match,
    40-59 Partial match, 0-39 Weak match.
    """
    if score >= 80:
        return "Strong match"
    elif score >= 60:
        return "Good match"
    elif score >= 40:
        return "Partial match"
    else:
        return "Weak match"


# ── App header ──────────────────────────────────────────────────────

st.set_page_config(page_title="CV Sorter Dashboard", layout="wide")
st.title("CV Sorter — Recruiter Dashboard")
st.caption("Powered by LLMs · Results loaded from cached JSON")

# ── Sidebar: file loader ───────────────────────────────────────────

with st.sidebar:
    st.header("Load Results")
    json_path = st.text_input(
        label="Results file path",
        value="results/ranked_output.json",
    )
    load_button = st.button("Load")

    st.divider()
    st.header("Filter Candidates")

    min_score_filter = st.slider(
        label="Minimum score",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        help="Hide candidates with overall score below this threshold",
    )

    skill_filter_input = st.text_input(
        label="Filter by skill (optional)",
        placeholder="e.g. Python",
        help="Show only candidates who scored >= 50 on this requirement",
    )

if load_button:
    try:
        with open(json_path, encoding="utf-8") as f:
            st.session_state["results"] = json.load(f)
    except FileNotFoundError:
        st.error(f"File not found: {json_path}")
    except json.JSONDecodeError:
        st.error("File is not valid JSON.")

if "results" not in st.session_state:
    st.info("Load a results file from the sidebar to get started.")
    st.stop()

results: dict = st.session_state["results"]

# ── Metadata bar ───────────────────────────────────────────────────

col1, col2, col3 = st.columns(3)
col1.metric(label="Job", value=results["job_title"])
col2.metric(label="Provider", value=f"{results['provider']} / {results['model']}")
col3.metric(label="Candidates", value=str(results["total_candidates"]))

st.divider()

# ── Candidate cards ────────────────────────────────────────────────

candidates: list[dict] = results["candidates"]

filtered_candidates = candidates

if min_score_filter > 0:
    filtered_candidates = [
        c for c in filtered_candidates
        if c["overall_score"] >= min_score_filter
    ]

if skill_filter_input.strip():
    skill_lower = skill_filter_input.strip().lower()

    def candidate_has_skill(candidate: dict) -> bool:
        for req in candidate["requirement_scores"]:
            if skill_lower in req["requirement"].lower():
                return req["score"] >= 50
        return False

    filtered_candidates = [
        c for c in filtered_candidates
        if candidate_has_skill(c)
    ]

if not filtered_candidates:
    st.warning(
        "No candidates match the current filters. "
        "Try lowering the minimum score or clearing the skill filter."
    )
else:
    st.caption(
        f"Showing {len(filtered_candidates)} of "
        f"{len(candidates)} candidates"
    )

for candidate in filtered_candidates:
    label = (
        f"#{candidate['rank']}  {candidate['filename']}  "
        f"— {candidate['overall_score']}/100  ·  {candidate['fit_label']}"
    )
    with st.expander(label):
        st.write("**Explanation:**", candidate["explanation"])
        st.divider()
        st.write("**Per-requirement breakdown:**")

        for req in candidate["requirement_scores"]:
            c1, c2, c3 = st.columns([3, 1, 5])
            with c1:
                st.write(req["requirement"])
            with c2:
                st.markdown(
                    score_badge(req["score"]),
                    unsafe_allow_html=True,
                )
            with c3:
                st.caption(req["evidence"])

# ── Interactive re-ranking ─────────────────────────────────────────

st.divider()
st.subheader("Interactive Re-ranking")
st.caption(
    "Adjust requirement weights and click Re-rank. "
    "No LLM calls — re-ranks using cached scores."
)

if not candidates:
    st.warning("No candidates loaded — cannot re-rank.")
else:
    requirement_names: list[str] = [
        rs["requirement"] for rs in candidates[0]["requirement_scores"]
    ]

    with st.container(border=True):
        for req_name in requirement_names:
            left, right = st.columns([2, 5])
            with left:
                st.write(req_name)
            with right:
                st.slider(
                    label=req_name,
                    min_value=0.0,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    key=f"weight_{req_name}",
                    label_visibility="collapsed",
                )

    rerank_button = st.button("Re-rank", type="primary")

    if rerank_button:
        weights: dict[str, float] = {
            req_name: st.session_state[f"weight_{req_name}"]
            for req_name in requirement_names
        }
        reranked = sorted(
            candidates,
            key=lambda c: compute_weighted_score(c, weights),
            reverse=True,
        )
        st.session_state["reranked_candidates"] = reranked
        st.session_state["rerank_weights"] = weights

    if "reranked_candidates" in st.session_state:
        weights = st.session_state.get("rerank_weights", {})
        st.write("**Re-ranked results:**")

        for new_rank, cand in enumerate(
            st.session_state["reranked_candidates"], start=1
        ):
            new_score = compute_weighted_score(cand, weights)
            new_fit_label = derive_fit_label(new_score)
            original_rank = cand["rank"]

            r1, r2, r3, r4 = st.columns([1, 4, 2, 3])
            with r1:
                st.write(f"**{new_rank}**")
            with r2:
                st.write(cand["filename"])
            with r3:
                st.markdown(
                    score_badge(new_score),
                    unsafe_allow_html=True,
                )
            with r4:
                st.write(new_fit_label)

            if new_rank != original_rank:
                st.caption(f"  (was #{original_rank})")
