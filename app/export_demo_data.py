"""Export the evaluated ML decisions used by the Streamlit demo.

Run from the project root with:
    python -m app.export_demo_data

The interface loads the exported table rather than retraining the model every
time a user changes a dropdown. Seed 6202 is used so the five held-out LLM
evaluation records are exact members of the displayed decision set.
"""

from __future__ import annotations

from pathlib import Path

from llm.run_explanation_eval import build_ml_decision_records

OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "demo_decisions_seed6202.csv"


def main() -> None:
    decisions = build_ml_decision_records(seed=6202)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(OUTPUT_PATH, index=False)
    print(
        f"Saved {len(decisions):,} decisions for "
        f"{decisions['sku_id'].nunique():,} SKUs to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()

