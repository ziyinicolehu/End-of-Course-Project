"""RestockIQ buyer-facing Streamlit demonstration interface."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# `streamlit run app/app.py` places app/ rather than the repository root at
# the front of sys.path on some installations. Add the root explicitly so the
# existing project packages are importable without installing the repo itself.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the sibling directly. Streamlit executes this file with app/ at the
# front of sys.path, where importing ``app.data_tools`` can otherwise confuse
# this file (app.py) with the app package.
from data_tools import (  # noqa: E402
    DECISIONS_PATH,
    FINAL_LLM_EVIDENCE_PATH,
    FINAL_EVALUATION_PATH,
    REVIEWED_EXPLANATIONS_PATH,
    decision_at,
    decision_display_values,
    live_run_summary,
    load_decisions,
    load_json,
    reviewed_record_for,
)
from llm.eval_explanation import evaluate_explanation  # noqa: E402
from llm.explain import DECISION_FIELDS, DEFAULT_MODEL, explain_decision  # noqa: E402


st.set_page_config(
    page_title="RestockIQ",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.8rem; padding-bottom: 3rem;}
      [data-testid="stMetric"] {
        background: #f7f9fc;
        border: 1px solid #e4e9f2;
        border-radius: 12px;
        padding: 14px 16px;
      }
      .status-box {
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0 1rem 0;
      }
      .reviewed {background: #edf8f1; border: 1px solid #9ad0ad;}
      .unreviewed {background: #fff8e6; border: 1px solid #e7c86e;}
      .footnote {color: #5d6778; font-size: 0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_decisions() -> pd.DataFrame:
    return load_decisions(DECISIONS_PATH)


@st.cache_data(show_spinner=False)
def get_final_evaluation() -> dict:
    return load_json(FINAL_EVALUATION_PATH)


@st.cache_data(show_spinner=False)
def get_reviewed_records() -> list[dict]:
    return load_json(REVIEWED_EXPLANATIONS_PATH)


@st.cache_data(show_spinner=False)
def get_final_llm_records() -> list[dict]:
    return load_json(FINAL_LLM_EVIDENCE_PATH)


def buyer_decision(decision: dict) -> dict:
    """Restrict a CSV row to the current V4 LLM decision contract."""
    return {field: decision[field] for field in DECISION_FIELDS}


def money(value: float | None) -> str:
    return "Not recorded" if value is None else f"${value:,.6f}"


def show_explanation(record: dict, reviewed: bool) -> None:
    response = record.get("parsed_response") or {}
    explanation = response.get("explanation")
    confidence = response.get("recommendation_confidence", "not available")
    if not explanation:
        st.error("No usable explanation was returned for this decision.")
        if record.get("api_error"):
            st.caption(record["api_error"].get("error_message", "API call failed."))
        elif record.get("parse_error"):
            st.caption(record["parse_error"])
        return

    if reviewed:
        passed = record.get("human_prose_accurate") is True
        css_class = "reviewed" if passed else "unreviewed"
        label = "Human-reviewed: passed" if passed else "Human-reviewed: limitation found"
        st.markdown(
            f'<div class="status-box {css_class}"><strong>{label}</strong></div>',
            unsafe_allow_html=True,
        )
    else:
        passed = record.get("deterministic_passes") is True
        css_class = "reviewed" if passed else "unreviewed"
        label = (
            "Automatic checks passed — wording has not been human-reviewed"
            if passed
            else "Automatic checks did not pass — do not rely on this explanation"
        )
        st.markdown(
            f'<div class="status-box {css_class}"><strong>{label}</strong></div>',
            unsafe_allow_html=True,
        )

    st.write(explanation)
    st.caption(f"Recommendation confidence: {str(confidence).title()}")
    if reviewed and record.get("human_review_notes"):
        st.info(record["human_review_notes"], icon="🔎")


def decision_explorer(api_key: str | None = None) -> None:
    decisions = get_decisions()
    reviewed_records = get_reviewed_records()

    st.subheader("Decision Explorer")
    st.write(
        "Choose any synthetic SKU to inspect its forecast, inventory position, "
        "and recommended order."
    )

    sku_ids = sorted(decisions["sku_id"].unique())
    default_sku = "FM-003" if "FM-003" in sku_ids else sku_ids[0]
    sku_id = st.selectbox(
        "SKU",
        sku_ids,
        index=sku_ids.index(default_sku),
        help=f"All {len(sku_ids)} synthetic SKUs are available.",
    )

    sku_rows = decisions[decisions["sku_id"] == sku_id]
    reviewed_weeks = {
        int(r["decision"]["week"])
        for r in reviewed_records
        if r.get("decision", {}).get("sku_id") == sku_id
    }
    available_weeks = sorted(sku_rows["week"].astype(int).unique(), reverse=True)
    default_week = next(iter(reviewed_weeks), available_weeks[0])

    def format_week(week: int) -> str:
        suffix = " · formally reviewed example" if week in reviewed_weeks else ""
        return f"Week {week}{suffix}"

    week = st.selectbox(
        "Decision week",
        available_weeks,
        index=available_weeks.index(default_week),
        format_func=format_week,
    )

    decision = decision_at(decisions, sku_id, week)
    display = decision_display_values(decision)
    reviewed_record = reviewed_record_for(reviewed_records, sku_id, week)
    decision_key = f"{sku_id}:{week}"
    live_record = st.session_state.get(f"live:{decision_key}")
    confidence_record = live_record or reviewed_record or {}
    confidence = (
        confidence_record.get("parsed_response") or {}
    ).get("recommendation_confidence", "Pending explanation")

    st.caption(
        f"Synthetic demo data · seed 6202 · {decision['pattern'].replace('_', ' ').title()}"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Next 4 weeks forecast", f"{decision['forecast_4_week_demand']:,.0f} units")
    c2.metric("Recommended order", f"{decision['order_quantity']:,.0f} units")
    c3.metric("Supplier lead time", f"{decision['supplier_lead_time_weeks']:,.0f} weeks")
    c4.metric("Recommendation confidence", str(confidence).title())

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown("#### Inventory picture")
        chart = pd.DataFrame(
            {
                "Units": [
                    display["physical_after_sales"],
                    display["incoming_stock"],
                    decision["required_stock"],
                    decision["order_quantity"],
                ]
            },
            index=[
                "Physical stock after this week's sales",
                "Already on the way",
                "Recommended stock target",
                "New order",
            ],
        )
        st.bar_chart(chart, horizontal=True, color="#2456A6")
        st.caption(
            "The stock target is a comparison point, not an additional stock category."
        )

    with right:
        st.markdown("#### Why this order?")
        if reviewed_record:
            show_explanation(reviewed_record, reviewed=True)
            st.caption("You can generate a fresh live response below for comparison.")
        else:
            st.markdown(
                '<div class="status-box unreviewed"><strong>No saved, human-reviewed '
                "explanation exists for this decision.</strong></div>",
                unsafe_allow_html=True,
            )

        has_key = bool(api_key)
        if not has_key:
            st.caption(
                "Enter your OpenRouter API key in the sidebar to enable live explanations."
            )
        if st.button(
            "Generate live explanation",
            type="primary",
            disabled=not has_key,
            use_container_width=True,
        ):
            with st.spinner("Generating and checking the explanation…"):
                try:
                    llm_decision = buyer_decision(decision)
                    explanation, metadata = explain_decision(
                        llm_decision,
                        live=True,
                        model=DEFAULT_MODEL,
                        api_key=api_key,
                    )
                    evaluation = evaluate_explanation(llm_decision, explanation)
                    st.session_state[f"live:{decision_key}"] = {
                        "decision": llm_decision,
                        "parsed_response": explanation,
                        "deterministic_passes": evaluation.passes,
                        "schema_valid": evaluation.schema_valid,
                        "grounded": evaluation.grounded,
                        "within_word_limit": evaluation.within_word_limit,
                        "model_served": metadata.model_served,
                        "elapsed_seconds": metadata.elapsed_seconds,
                        "total_tokens": metadata.total_tokens,
                        "cost_usd": metadata.cost_usd,
                    }
                    st.rerun()
                except Exception as exc:
                    st.session_state[f"live:{decision_key}"] = {
                        "parsed_response": None,
                        "deterministic_passes": False,
                        "api_error": {"error_message": str(exc)},
                    }
        live_record = st.session_state.get(f"live:{decision_key}")
        if live_record:
            st.markdown("##### Newly generated live response")
            show_explanation(live_record, reviewed=False)
            if live_record.get("model_served"):
                st.caption(
                    f"Model: {live_record['model_served']} · "
                    f"Latency: {live_record.get('elapsed_seconds', 0):.2f}s · "
                    f"Tokens: {live_record.get('total_tokens', 'not recorded')} · "
                    f"Cost: {money(live_record.get('cost_usd'))}"
                )

    with st.expander("Show the calculation"):
        st.write(
            f"The model forecasts **{decision['forecast_4_week_demand']:,.1f} units** "
            f"over four weeks, or **{decision['weekly_predicted_demand']:,.1f} units per week**."
        )
        st.write(
            f"The protection period is the {decision['supplier_lead_time_weeks']:,.0f}-week "
            f"supplier lead time plus one review week: "
            f"**{decision['protection_period_weeks']:,.0f} weeks**."
        )
        st.write(
            f"Safety stock is **{decision['safety_stock']:,.1f} units**, producing a "
            f"recommended stock target of **{decision['required_stock']:,.1f} units**."
        )
        st.write(
            f"Stock after this week's sales plus orders already on the way totals "
            f"**{decision['inventory_position']:,.1f} units**. The policy therefore "
            f"recommends ordering **{decision['order_quantity']:,.0f} units**."
        )

    if decision["supplier_lead_time_weeks"] > 4:
        st.warning(
            "This supplier's lead time is longer than the four-week forecast horizon. "
            "Part of the stock target therefore extrapolates the forecast beyond the "
            "period directly predicted.",
            icon="⚠️",
        )


def evaluation_results() -> None:
    result = get_final_evaluation()
    forecast = pd.DataFrame(result["forecast"]["patterns"])
    reorder = result["reorder"]
    llm = result["llm"]
    records = get_final_llm_records()

    st.subheader("Evaluation Evidence")
    st.write("The forecast, reorder policy, and explanation layer are evaluated separately.")

    st.markdown("#### Forecast accuracy")
    pattern_order = [
        "fast_moving",
        "slow_moving",
        "seasonal",
        "intermittent",
        "promotion_driven",
    ]
    chart = forecast.set_index("pattern").reindex(pattern_order)[
        ["baseline_wmape", "ml_wmape"]
    ].rename(
        columns={"baseline_wmape": "Moving-average baseline", "ml_wmape": "ML model"}
    ) * 100
    chart.index = [value.replace("_", " ").title() for value in chart.index]
    st.bar_chart(
        chart,
        y_label="WMAPE (%)",
        color=["#7CBCE8", "#0B6ECF"],
        sort=False,
        stack=False,
    )
    st.caption("Lower WMAPE is better. Intermittent demand remains the hardest pattern.")

    st.markdown("#### Simulated business outcome")
    b1, b2, b3 = st.columns(3)
    b1.metric(
        "Lost-demand reduction",
        f"{reorder['lost_demand_reduction_pct']:.1f}%",
        f"Target ≥ {reorder['lost_demand_reduction_threshold_pct']:.0f}%",
    )
    b2.metric(
        "Excess-inventory change",
        f"{reorder['excess_inventory_increase_pct']:.1f}%",
        f"{reorder['excess_inventory_increase_pct']:.1f}% vs baseline",
        delta_color="inverse",
        help=(
            "Passed: the locked limit allowed no more than a "
            f"{reorder['excess_inventory_increase_threshold_pct']:.0f}% increase."
        ),
    )
    b3.metric("Locked success criteria", "Passed" if reorder["passes"] else "Not passed")
    st.caption("These are simulated results on synthetic data, not measured commercial ROI.")

    st.markdown("#### LLM explanation quality")
    l1, l2, l3 = st.columns(3)
    l1.metric("Automatic checks", f"{llm['deterministic_passes']}/{llm['total_live_decisions']}")
    l2.metric("Strict human review", f"{llm['human_review_passes']}/{llm['total_live_decisions']}")
    l3.metric("Human-review failures", str(llm["human_review_failures"]))

    usage = live_run_summary(records)
    if usage["total_decisions"] != llm["total_live_decisions"]:
        st.error(
            "The saved usage evidence does not match the displayed evaluation run."
        )
    st.caption(
        f"Held-out live run: {usage['total_tokens']:,} tokens · "
        f"{money(usage['total_cost_usd'])} total · "
        f"{usage['average_latency_seconds']:.2f}s average latency per explanation."
    )

    failed = llm.get("failed_decisions", [])
    if failed:
        st.warning(
            f"Documented limitation — {failed[0]['sku_id']}: {failed[0]['limitation']}",
            icon="⚠️",
        )


def how_it_works() -> None:
    st.subheader("How RestockIQ Works")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown("### 1\n**Synthetic history**\n\nSeeded Python generates weekly sales from reviewed parameter ranges.")
    c2.markdown("### 2\n**ML forecast**\n\nThe model predicts demand over the next four weeks.")
    c3.markdown("### 3\n**Reorder rule**\n\nDeterministic Python calculates the stock target and order quantity.")
    c4.markdown("### 4\n**LLM explanation**\n\nA model translates the fixed decision into buyer-friendly language.")
    c5.markdown("### 5\n**Human review**\n\nA buyer remains responsible before any real purchase order is placed.")

    st.divider()
    st.markdown("#### Deliberate boundaries")
    st.markdown(
        """
        - The LLM does **not** calculate quantities or modify the reorder decision.
        - RestockIQ does **not** place purchase orders automatically.
        - The demonstration uses synthetic data and does not claim measured real-world ROI.
        - RAG, fine-tuning and an autonomous agent were not added because this is a fixed,
          structured workflow with no document-retrieval requirement.
        - Deployment with real data would require integration, monitoring, drift checks,
          access controls and buyer approval.
        """
    )


st.title("📦 RestockIQ")
st.caption("AI-assisted reorder decision support for wholesale apparel buyers")

page = st.sidebar.radio(
    "Navigate",
    ["Decision Explorer", "Evaluation Evidence", "How It Works"],
)
st.sidebar.divider()
st.sidebar.markdown("**Optional live explanations**")
entered_api_key = st.sidebar.text_input(
    "OpenRouter API key",
    type="password",
    placeholder="Paste key here",
    help=(
        "The key stays in this running app's memory only. It is not written "
        "to the repository or saved result files."
    ),
)
api_key = entered_api_key.strip() or os.environ.get("OPENROUTER_API_KEY")
if api_key:
    st.sidebar.success("Live explanations enabled")
else:
    st.sidebar.caption("Saved reviewed explanations work without a key.")
st.sidebar.divider()
st.sidebar.caption(
    "Course demonstration only · synthetic data · no purchase orders are placed"
)

if page == "Decision Explorer":
    decision_explorer(api_key=api_key)
elif page == "Evaluation Evidence":
    evaluation_results()
else:
    how_it_works()
