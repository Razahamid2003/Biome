"""Run custom Great Expectations validation and build Data Docs.

- Loads the custom expectation plugin from gx/plugins/expectations/...
- Validates a CSV with a column-pair difference expectation
- Saves an Expectation Suite
- Runs a SimpleCheckpoint (with COMPLETE result_format)
- Builds & opens Data Docs

Great Expectations version: 0.18.21
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import great_expectations as gx
from great_expectations.checkpoint import SimpleCheckpoint


# ---------------- USER CONFIG ----------------
CSV_FILENAME = "Task.csv"
COLUMN_A_NAME = "Value_A"
COLUMN_B_NAME = "Value_B"
MIN_THRESHOLD: Optional[float] = 0
MAX_THRESHOLD: Optional[float] = 10
SORT = True
SORT_KEY = "Value_A"
IGNORE_ROW_IF = "both_values_are_missing"
MOSTLY = 0.8
SUITE_NAME = "task2_custom_diff_suite"
RUNTIME_RESULT_FORMAT: Dict[str, Any] = {
    "result_format": "COMPLETE",
    "include_unexpected_rows": True,
}
# ------------------------------------------------


def _die(msg: str) -> None:
    """Exit the program with an error message."""
    print(f"ERROR: {msg}")
    raise SystemExit(1)


def _load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        _die(f"CSV not found at: {csv_path}")
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:
        _die(f"Failed to read CSV: {exc}")
    return pd.DataFrame()


def _import_plugin_by_path(plugin_path: Path) -> None:
    """Import the custom expectation module by file path (execs the module for side effects)."""
    if not plugin_path.is_file():
        _die(
            "Custom expectation file not found.\n"
            f"Expected here:\n  {plugin_path}\n"
            "Move your file to this path and re-run."
        )
    spec = spec_from_file_location("expect_column_pair_values_diff_within_range", str(plugin_path))
    if spec is None or spec.loader is None:
        _die("Failed to load plugin spec.")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)


def _validate_user_inputs(df: pd.DataFrame) -> None:
    missing_cols = [c for c in (COLUMN_A_NAME, COLUMN_B_NAME) if c not in df.columns]
    if missing_cols:
        _die(f"Missing columns in {CSV_FILENAME}: {missing_cols}")
    if SORT and SORT_KEY not in (COLUMN_A_NAME, COLUMN_B_NAME):
        _die(f"SORT_KEY must be {COLUMN_A_NAME} or {COLUMN_B_NAME} when SORT=True.")
    if MIN_THRESHOLD is None and MAX_THRESHOLD is None:
        _die("Provide at least one of MIN_THRESHOLD or MAX_THRESHOLD.")
    if (
        MIN_THRESHOLD is not None
        and MAX_THRESHOLD is not None
        and float(MIN_THRESHOLD) > float(MAX_THRESHOLD)
    ):
        _die("MIN_THRESHOLD cannot be greater than MAX_THRESHOLD.")


def main() -> None:
    print(f"Starting script... (GX: {gx.__version__} )")

    # ---------- Load CSV ----------
    script_dir = Path(__file__).resolve().parent
    csv_path = script_dir / CSV_FILENAME
    df = _load_csv(csv_path)

    _validate_user_inputs(df)
    print("Inputs validated. Proceeding...")

    # ---------- Data Context ----------
    context: Any = gx.get_context()
    print("GX Context root:", context.root_directory)

    # ---------- Load the plugin by file path ----------
    plugin_path = Path(context.root_directory) / "plugins" / "expectations" / (
        "expect_column_pair_values_diff_within_range.py"
    )
    _import_plugin_by_path(plugin_path)

    # ---------- Datasource & Validator ----------
    datasource = context.sources.add_or_update_pandas(name="pandas")
    validator = datasource.read_dataframe(df, asset_name="task_asset")

    if not hasattr(validator, "expect_column_pair_values_diff_within_range"):
        _die(
            "Custom expectation not registered on Validator. "
            "Double-check the plugin file content and re-run."
        )

    # ---------- Ad-hoc run ----------
    min_str = str(MIN_THRESHOLD) if MIN_THRESHOLD is not None else "-inf"
    max_str = str(MAX_THRESHOLD) if MAX_THRESHOLD is not None else "inf"
    sort_key_str = SORT_KEY if SORT else "None"

    print(
        f"\n--- Validating: abs({COLUMN_A_NAME}-{COLUMN_B_NAME}) within "
        f"[{min_str}, {max_str}] (sort={SORT}, key={sort_key_str}) ---"
    )

    result = validator.expect_column_pair_values_diff_within_range(
        column_A=COLUMN_A_NAME,
        column_B=COLUMN_B_NAME,
        min_threshold=MIN_THRESHOLD,
        max_threshold=MAX_THRESHOLD,
        sort=SORT,
        sort_key=SORT_KEY if SORT else None,
        ignore_row_if=IGNORE_ROW_IF,
        mostly=MOSTLY,
        result_format=RUNTIME_RESULT_FORMAT["result_format"],
        include_unexpected_rows=RUNTIME_RESULT_FORMAT["include_unexpected_rows"],
    )
    print(result)
    print("\nAd-hoc validation success:", result.success)

    # ---------- Save the Expectation Suite ----------
    validator.expectation_suite_name = SUITE_NAME
    validator.save_expectation_suite(discard_failed_expectations=False)
    suite_name = validator.expectation_suite_name
    print(f"Saved Expectation Suite: {suite_name}")

    # ---------- Checkpoint ----------
    batch_request = validator.active_batch.batch_request
    checkpoint = SimpleCheckpoint(
        name="task2_custom_diff_checkpoint",
        data_context=context,
        validations=[{"batch_request": batch_request, "expectation_suite_name": suite_name}],
    )

    cp_result = checkpoint.run(result_format=RUNTIME_RESULT_FORMAT)
    print("\nCheckpoint run success:", cp_result.get("success"))

    # ---------- Build & Open Data Docs ----------
    context.build_data_docs()
    try:
        context.open_data_docs()
        print("Opened Data Docs.")
    except Exception as ex:
        print("Data Docs could not be opened automatically:", ex)
        print("Open the site from:", Path(context.root_directory) / "data_docs")


if __name__ == "__main__":
    main()
