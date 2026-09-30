"""RestockIQ eval harness -- runs the full three-layer eval story with
one command, prints one consolidated report, and automatically saves
BOTH a human-readable copy of that report and a machine-readable
summary. Nobody needs to copy anything out of Terminal by hand.

Ties together the three separately-evaluable pieces this project keeps
apart on purpose (see README.md's opening line): the forecast model, the
reorder policy, and the LLM explanation layer. This script does not
reimplement or re-derive any of their locked logic -- it only calls each
layer's own run()/load function and lays the results out side by side.

    1. Forecast (models.run_forecast_eval.run): ML vs. the moving-average
       baseline, per-pattern WMAPE, including patterns where ML loses.
       Diagnostic -- WMAPE is not itself a pass/fail gate.
    2. Reorder (reorder.run_reorder_eval.run_detailed): the two locked
       business metrics from CONTRACT.md -- lost-demand reduction
       (target >=10%) and excess-inventory increase (target <=5%) --
       both must hold at once for the reorder policy to "pass". Uses
       run_detailed() rather than run() so the raw baseline/ML totals
       (not just the two percentages) are available for the saved JSON,
       without running the simulation twice or inventing a second
       formula for numbers evaluate_success() already computes.
    3. LLM explanation layer (llm.run_explanation_eval.load_results +
       print_combined_report): the combined deterministic + human-review
       status of a SPECIFIED saved evidence file -- by default, the
       final held-out evaluation. This harness NEVER makes a live API
       call itself -- only llm.run_explanation_eval's own --live flag
       does that, run by hand and reviewed by a human afterward. This
       script only reads the evidence file it's given; it does not
       search for or infer "the most recent" one by file date, so the
       loaded file is always named explicitly, both on screen and in
       the saved outputs.

These three are reported SEPARATELY here, never blended into one score.
CONTRACT.md defines "success" only for the reorder policy's two business
metrics; forecast WMAPE is diagnostic; the LLM explanation layer has its
own separate bar under EXPLANATION_CONTRACT.md, decided by human review,
not a number this script could compute. Combining all three into a
single verdict would hide which piece is actually driving any given
result -- exactly what CONTRACT.md was locked to avoid.

The LLM evidence file does not record which seed it was generated with
(see llm/run_explanation_eval.py's saved record schema), so --llm-seed
is metadata ASSERTED by the caller for the file named by --llm-results,
not something this script reads out of that file. Its default (6204)
matches the default --llm-results file (the V4 25-case held-out
evaluation, documented in LLM_EXPLANATION_DESIGN.md) --
override both together if you point this at a different saved file.

Usage:
    # forecast + reorder (live, fast, no network) + LLM (reads the
    # SPECIFIED saved evidence file; makes no API call itself). Always
    # saves evals/results/final_evaluation.json and
    # evals/results/eval_harness_report.txt from this same run.
    python3 -m evals.run_all [--seed 6201]

    # point at a specific saved LLM results file (update --llm-seed to
    # match whatever seed that file actually used, if not 6201/6204)
    python3 -m evals.run_all --seed 6201 --llm-results evals/results/llm_live_evaluation_v3_comparison.json --llm-seed 6201

    # forecast + reorder only, no LLM results file needed
    python3 -m evals.run_all --skip-llm
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd

from evals.contract import SuccessResult
from llm.run_explanation_eval import combined_status, load_results, print_combined_report
from models.run_forecast_eval import run as run_forecast_eval
from reorder.run_reorder_eval import ReorderEvalDetail, run_detailed as run_reorder_eval_detailed

DEFAULT_LLM_RESULTS_PATH = Path("evals/results/llm_live_evaluation_v4_seed6204.json")
DEFAULT_LLM_SEED = 6204  # V4 25-case held-out evaluation -- see module docstring
DEFAULT_JSON_OUTPUT_PATH = Path("evals/results/final_evaluation.json")
DEFAULT_REPORT_OUTPUT_PATH = Path("evals/results/eval_harness_report.txt")

LOST_DEMAND_REDUCTION_THRESHOLD_PCT = 10.0
EXCESS_INVENTORY_INCREASE_THRESHOLD_PCT = 5.0


class _Tee:
    """Writes everything to every stream it's given. Used so the SAME
    run's print statements go to both the real terminal and an
    in-memory buffer at once -- the buffer becomes the saved text
    report, so the two can never silently disagree (there is no second
    invocation that could produce different numbers)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def run(
    seed: int = 6201,
    llm_results_path: Path | None = DEFAULT_LLM_RESULTS_PATH,
    llm_seed: int | None = DEFAULT_LLM_SEED,
    save_json_path: Path | None = DEFAULT_JSON_OUTPUT_PATH,
    save_report_path: Path | None = DEFAULT_REPORT_OUTPUT_PATH,
) -> dict:
    """Runs the forecast eval and the reorder eval live -- both are pure
    synthetic-data generation plus local computation, no network call,
    so re-running them here is fast and makes nothing stale. Loads the
    LLM explanation layer's evidence from llm_results_path (no live call
    made here -- see module docstring). Prints all three reports plus
    one combined summary to the terminal, and -- unless the
    corresponding path is None -- ALSO writes the exact same printed
    report to save_report_path and a machine-readable summary to
    save_json_path, from this one run.

    Passing llm_results_path=None skips the LLM section entirely (and
    the saved JSON's "llm" section notes that honestly rather than
    inventing a status).

    Returns a dict with each layer's raw result, for anything that
    wants to consume it programmatically (e.g. tests) instead of
    parsing printed text.
    """
    buffer = StringIO()
    real_stdout = sys.stdout
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sys.stdout = _Tee(real_stdout, buffer)
    try:
        print("=" * 70)
        print(f"RestockIQ -- combined eval harness (generated {generated_utc} UTC)")
        print("=" * 70)
        print(
            f"Forecast and reorder were run live, just now, using seed={seed}. "
            "No live API call was made by this harness at any point."
        )

        print("\n[1/3] FORECAST EVAL (models.run_forecast_eval, seed=%d)\n" % seed)
        forecast_comparison = run_forecast_eval(seed=seed)

        print(
            "\n[2/3] REORDER EVAL -- locked business success metric "
            "(reorder.run_reorder_eval, seed=%d)\n" % seed
        )
        reorder_detail = run_reorder_eval_detailed(seed=seed)

        llm_records = None
        if llm_results_path is not None:
            print(
                "\n[3/3] LLM EXPLANATION EVAL -- specified saved evidence "
                "(final held-out evaluation, unless a different file was "
                "given), NO live call made\n"
            )
            print(f"  evidence file: {llm_results_path}")
            if llm_seed is not None:
                print(
                    f"  evidence seed: {llm_seed} (asserted for this file -- "
                    "not read from it; the saved evidence does not record "
                    "which seed produced it)"
                )
            if Path(llm_results_path).exists():
                llm_records = load_results(llm_results_path)
                print()
                print_combined_report(llm_records)
            else:
                print(
                    f"  No results file found at {llm_results_path} -- run "
                    "llm.run_explanation_eval --live yourself, review it by "
                    "hand, then point --llm-results at the saved file. "
                    "Skipping this section rather than claiming a status "
                    "that doesn't exist."
                )
        else:
            print("\n[3/3] LLM EXPLANATION EVAL -- skipped (--skip-llm)\n")

        _print_summary(forecast_comparison, reorder_detail.result, llm_records)
    finally:
        sys.stdout = real_stdout

    report_text = buffer.getvalue()

    if save_report_path is not None:
        save_report_path = Path(save_report_path)
        save_report_path.parent.mkdir(parents=True, exist_ok=True)
        save_report_path.write_text(report_text)

    consolidated = _build_consolidated_result(
        generated_utc=generated_utc,
        seed=seed,
        llm_seed=llm_seed,
        llm_results_path=llm_results_path,
        forecast_comparison=forecast_comparison,
        reorder_detail=reorder_detail,
        llm_records=llm_records,
    )
    if save_json_path is not None:
        save_json_path = Path(save_json_path)
        save_json_path.parent.mkdir(parents=True, exist_ok=True)
        save_json_path.write_text(json.dumps(consolidated, indent=2) + "\n")

    return {
        "forecast_comparison": forecast_comparison,
        "reorder_result": reorder_detail.result,
        "reorder_detail": reorder_detail,
        "llm_records": llm_records,
        "consolidated_result": consolidated,
    }


def _print_summary(
    forecast_comparison: pd.DataFrame,
    reorder_result: SuccessResult,
    llm_records: list[dict] | None,
) -> None:
    print()
    print("=" * 70)
    print(
        "SUMMARY -- each layer against its own locked contract; "
        "never blended into one score"
    )
    print("=" * 70)

    n_patterns = len(forecast_comparison)
    n_ml_wins = int(forecast_comparison["ml_beats_baseline"].sum())
    print(
        f"Forecast:  ML beats the moving-average baseline on "
        f"{n_ml_wins}/{n_patterns} demand patterns (diagnostic -- WMAPE "
        "is not itself a pass/fail gate; see the table above)."
    )

    print(
        f"Reorder:   lost_demand_reduction="
        f"{reorder_result.lost_demand_reduction_pct:.2f}% (target >=10%), "
        f"excess_inventory_increase="
        f"{reorder_result.excess_inventory_increase_pct:.2f}% (target "
        f"<=5%) -- PASSES: {reorder_result.passes}"
    )

    if llm_records is None:
        print("LLM:       no saved evaluation evidence loaded for this run.")
    else:
        live_records = [r for r in llm_records if r.get("mode") == "live"]
        if not live_records:
            print("LLM:       loaded results contain no live-mode records.")
        else:
            n_total = len(live_records)
            n_pass = sum(1 for r in live_records if combined_status(r).startswith("PASS"))
            print(
                f"LLM:       {n_pass}/{n_total} live decisions passed "
                "deterministic + strict human review on the loaded "
                "evaluation (see LLM_EXPLANATION_DESIGN.md for the full "
                "account, including any documented limitations)."
            )

    print()
    print(
        "Each layer above is graded against its own locked contract "
        "(CONTRACT.md for forecast/reorder, EXPLANATION_CONTRACT.md for "
        "the LLM layer) and reported on its own terms -- this harness "
        "does not combine them into a single pass/fail verdict."
    )


def _summarize_llm_records(llm_records: list[dict]) -> dict:
    """Derives every LLM summary figure from the records themselves
    (via combined_status(), the module's single source of truth for a
    record's final status) rather than hard-coding counts -- so this
    stays correct if the evidence file it's pointed at ever changes,
    the same reason README.md states test counts as 'all non-live tests
    pass' rather than a number that can go stale."""
    live_records = [r for r in llm_records if r.get("mode") == "live"]

    n_api_error = sum(1 for r in live_records if r.get("api_error") is not None)
    n_deterministic_passes = sum(1 for r in live_records if r.get("deterministic_passes"))
    n_human_pass = sum(1 for r in live_records if r.get("human_prose_accurate") is True)
    n_human_fail = sum(1 for r in live_records if r.get("human_prose_accurate") is False)
    statuses = {r.get("sku_id"): combined_status(r) for r in live_records}
    n_pending = sum(1 for s in statuses.values() if s.startswith("PENDING"))
    n_combined_pass = sum(1 for s in statuses.values() if s.startswith("PASS"))
    n_combined_fail = sum(1 for s in statuses.values() if s.startswith("FAIL"))

    failed_decisions = [
        {
            "sku_id": r.get("sku_id"),
            "limitation": r.get("human_review_notes") or None,
        }
        for r in live_records
        if combined_status(r).startswith("FAIL (human")
    ]

    return {
        "total_live_decisions": len(live_records),
        "api_errors": n_api_error,
        "deterministic_passes": n_deterministic_passes,
        "human_review_passes": n_human_pass,
        "human_review_failures": n_human_fail,
        "pending_human_reviews": n_pending,
        "combined_passes": n_combined_pass,
        "combined_failures": n_combined_fail,
        "failed_skus": [d["sku_id"] for d in failed_decisions],
        "failed_decisions": failed_decisions,
    }


def _build_consolidated_result(
    generated_utc: str,
    seed: int,
    llm_seed: int | None,
    llm_results_path: Path | None,
    forecast_comparison: pd.DataFrame,
    reorder_detail: ReorderEvalDetail,
    llm_records: list[dict] | None,
) -> dict:
    """Builds the machine-readable summary as plain JSON-compatible
    values only (str/int/float/bool/None/list/dict) -- every pandas/
    NumPy value is explicitly converted, never passed through as-is,
    since e.g. numpy.bool_/numpy.float64 are not valid input to
    json.dumps without a custom encoder. Contains no secret or
    environment-variable value anywhere -- only already-public run
    parameters (seeds, file paths) and evaluation results."""
    forecast_patterns = [
        {
            "pattern": str(row["pattern"]),
            "n_test_rows": int(row["n_test_rows"]),
            "baseline_wmape": float(row["baseline_wmape"]),
            "ml_wmape": float(row["ml_wmape"]),
            "ml_beats_baseline": bool(row["ml_beats_baseline"]),
        }
        for _, row in forecast_comparison.iterrows()
    ]

    result = reorder_detail.result
    reorder_section = {
        "baseline_total_lost_demand": float(reorder_detail.baseline_total_lost_demand),
        "ml_total_lost_demand": float(reorder_detail.ml_total_lost_demand),
        "lost_demand_reduction_pct": float(result.lost_demand_reduction_pct),
        "lost_demand_reduction_threshold_pct": LOST_DEMAND_REDUCTION_THRESHOLD_PCT,
        "baseline_mean_excess_inventory": float(reorder_detail.baseline_mean_excess_inventory),
        "ml_mean_excess_inventory": float(reorder_detail.ml_mean_excess_inventory),
        "excess_inventory_increase_pct": float(result.excess_inventory_increase_pct),
        "excess_inventory_increase_threshold_pct": EXCESS_INVENTORY_INCREASE_THRESHOLD_PCT,
        "passes": bool(result.passes),
    }

    if llm_records is None:
        llm_section = {
            "evidence_source": str(llm_results_path) if llm_results_path is not None else None,
            "evidence_seed": llm_seed,
            "loaded": False,
            "note": "No LLM evidence was loaded for this run (skipped, or the specified file did not exist).",
        }
    else:
        llm_section = {
            "evidence_source": str(llm_results_path),
            "evidence_seed": llm_seed,
            "loaded": True,
            **_summarize_llm_records(llm_records),
        }

    return {
        "generated_utc": generated_utc,
        "run_info": {
            "forecast_and_reorder_seed": seed,
            "llm_evaluation_seed": llm_seed,
            "llm_evidence_source": str(llm_results_path) if llm_results_path is not None else None,
            "live_api_call_made": False,
        },
        "forecast": {
            "patterns": forecast_patterns,
            "n_patterns_ml_won": sum(1 for p in forecast_patterns if p["ml_beats_baseline"]),
            "n_patterns_total": len(forecast_patterns),
        },
        "reorder": reorder_section,
        "llm": llm_section,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=6201)
    parser.add_argument(
        "--llm-results",
        type=Path,
        default=DEFAULT_LLM_RESULTS_PATH,
        help=(
            "Path to a saved LLM explanation eval results file (produced "
            "by llm.run_explanation_eval --live --save, reviewed by "
            "hand). Never triggers a live call itself."
        ),
    )
    parser.add_argument(
        "--llm-seed",
        type=int,
        default=DEFAULT_LLM_SEED,
        help=(
            "The seed --llm-results was generated with. Asserted by you, "
            "not read from the file (it isn't stored there) -- update "
            "this if you point --llm-results at a file from a different "
            "seed."
        ),
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip the LLM explanation section entirely (forecast + reorder only).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_JSON_OUTPUT_PATH,
        help="Where to save the machine-readable consolidated summary.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_REPORT_OUTPUT_PATH,
        help="Where to save the human-readable text report (identical to what's printed).",
    )
    args = parser.parse_args()
    run(
        seed=args.seed,
        llm_results_path=None if args.skip_llm else args.llm_results,
        llm_seed=None if args.skip_llm else args.llm_seed,
        save_json_path=args.output_json,
        save_report_path=args.output_report,
    )
