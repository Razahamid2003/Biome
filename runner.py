"""Run Great Expectations validation and build Data Docs (GX 0.18.21)."""

from __future__ import annotations

import json
from importlib.abc import Loader
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Tuple, cast

import great_expectations as gx
import pandas as pd
from great_expectations.checkpoint import SimpleCheckpoint

CSV_FILENAME = "Task.csv"
CONFIG_FILENAME = "expectation_config.json"

RUNTIME_RESULT_FORMAT: Dict[str, Any] = {
    "result_format": "COMPLETE",
    "include_unexpected_rows": True,
}


def _die(msg: str) -> NoReturn:
    print(f"ERROR: {msg}")
    raise SystemExit(1)


def _looks_like_numeric_header(cols: List[Any]) -> bool:
    """Heuristic: header row seems numeric (typical when CSV has no header)."""
    if not cols:
        return False
    try:
        return all(str(c).strip() != "" and float(str(c).strip()) is not None for c in cols)
    except Exception:
        return False


def _load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        _die(f"CSV not found at: {csv_path}")
    try:
        # First attempt: assume header row exists
        df = pd.read_csv(csv_path, skipinitialspace=True)
        # If the inferred header looks numeric, re-read as headerless and name columns
        if _looks_like_numeric_header(list(df.columns)):
            df = pd.read_csv(csv_path, header=None, skipinitialspace=True)
            df.columns = [f"column_{i+1}" for i in range(df.shape[1])]
        return df
    except Exception as exc:
        _die(f"Failed to read CSV: {exc}")


def _load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.is_file():
        _die(f"Expectation config JSON not found at: {config_path}")
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _die(f"Failed to parse JSON: {exc}")

    min_th = cfg.get("min_threshold")
    max_th = cfg.get("max_threshold")
    if min_th is None and max_th is None:
        _die("Provide at least one of 'min_threshold' or 'max_threshold' in JSON.")
    if min_th is not None and not isinstance(min_th, (int, float)):
        _die("'min_threshold' must be numeric or null.")
    if max_th is not None and not isinstance(max_th, (int, float)):
        _die("'max_threshold' must be numeric or null.")
    if min_th is not None and max_th is not None and float(min_th) > float(max_th):
        _die("'min_threshold' cannot be greater than 'max_threshold'.")

    sort = bool(cfg.get("sort", False))
    sort_key = cfg.get("sort_key")

    ignore_row_if = cfg.get("ignore_row_if", "both_values_are_missing")
    allowed = {"both_values_are_missing", "either_value_is_missing", "neither"}
    if ignore_row_if not in allowed:
        _die(
            "'ignore_row_if' must be one of: "
            "both_values_are_missing, either_value_is_missing, neither."
        )

    mostly = cfg.get("mostly", 1.0)
    if not isinstance(mostly, (int, float)) or not (0.0 <= float(mostly) <= 1.0):
        _die("'mostly' must be a number between 0 and 1.")

    suite_name = cfg.get("suite_name", "task_custom_diff_suite")
    if not isinstance(suite_name, str) or not suite_name:
        _die("'suite_name' must be a non-empty string.")

    col_a = cfg.get("column_A")  # optional
    col_b = cfg.get("column_B")  # optional
    if col_a is not None and (not isinstance(col_a, str) or not col_a):
        _die("'column_A' must be a non-empty string if provided.")
    if col_b is not None and (not isinstance(col_b, str) or not col_b):
        _die("'column_B' must be a non-empty string if provided.")

    return {
        "column_A": col_a,
        "column_B": col_b,
        "min_threshold": min_th,
        "max_threshold": max_th,
        "sort": sort,
        "sort_key": sort_key,
        "ignore_row_if": ignore_row_if,
        "mostly": float(mostly),
        "suite_name": suite_name,
    }


def _resolve_columns(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[str, str]:
    """Return (col_A, col_B) with fallbacks for missing names."""
    ca = cfg.get("column_A")
    cb = cfg.get("column_B")

    if isinstance(ca, str) and isinstance(cb, str):
        missing = [c for c in (ca, cb) if c not in df.columns]
        if not missing:
            return ca, cb
        print(
            f"Warning: configured columns not found {missing}; "
            "falling back to auto-select numeric columns."
        )

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 2:
        a, b = str(numeric_cols[0]), str(numeric_cols[1])
        print(f"Auto-selected numeric columns: {a}, {b}")
        return a, b

    if len(df.columns) >= 2:
        a, b = str(df.columns[0]), str(df.columns[1])
        print(f"Auto-selected first columns: {a}, {b}")
        return a, b

    _die("CSV must contain at least two columns to run this expectation.")


def _import_plugin_by_path(plugin_path: Path) -> None:
    if not plugin_path.is_file():
        _die(f"Custom expectation not found at: {plugin_path}")
    spec = spec_from_file_location(
        "expect_column_pair_values_diff_within_range",
        str(plugin_path),
    )
    if spec is None or spec.loader is None:
        _die("Failed to load plugin spec.")
    spec_typed = cast(ModuleSpec, spec)
    loader = cast(Loader, spec_typed.loader)
    module = module_from_spec(spec_typed)
    loader.exec_module(module)


def main() -> None:
    print(f"Starting script... (GX: {gx.__version__})")

    base = Path(__file__).resolve().parent
    df = _load_csv(base / CSV_FILENAME)
    cfg = _load_config(base / CONFIG_FILENAME)
    col_a, col_b = _resolve_columns(df, cfg)

    if cfg["sort"]:
        if cfg["sort_key"] not in (col_a, col_b):
            print(
                "Warning: 'sort' is true but 'sort_key' is missing or doesn't match "
                f"resolved columns ({col_a}, {col_b}). Defaulting sort_key to '{col_a}'."
            )
            cfg["sort_key"] = col_a

    context: Any = gx.get_context()
    plugin_path = (
        Path(context.root_directory)
        / "plugins"
        / "expectations"
        / "expect_column_pair_values_diff_within_range.py"
    )
    _import_plugin_by_path(plugin_path)

    datasource = context.sources.add_or_update_pandas(name="pandas")
    validator = datasource.read_dataframe(df, asset_name="task_asset")

    if not hasattr(validator, "expect_column_pair_values_diff_within_range"):
        _die("Custom expectation not registered on Validator. Check the plugin file and re-run.")

    print(
        f"Validating: abs({col_a} - {col_b}) within "
        f"[{cfg['min_threshold'] if cfg['min_threshold'] is not None else '-inf'}, "
        f"{cfg['max_threshold'] if cfg['max_threshold'] is not None else 'inf'}] "
        f"(sort={cfg['sort']}, key={cfg['sort_key'] if cfg['sort'] else 'None'})"
    )

    result = validator.expect_column_pair_values_diff_within_range(
        column_A=col_a,
        column_B=col_b,
        min_threshold=cfg["min_threshold"],
        max_threshold=cfg["max_threshold"],
        sort=cfg["sort"],
        sort_key=cfg["sort_key"],
        ignore_row_if=cfg["ignore_row_if"],
        mostly=cfg["mostly"],
        result_format=RUNTIME_RESULT_FORMAT["result_format"],
        include_unexpected_rows=RUNTIME_RESULT_FORMAT["include_unexpected_rows"],
    )
    print("\nAd-hoc validation success:", result.success)

    validator.expectation_suite_name = cfg["suite_name"]
    validator.save_expectation_suite(discard_failed_expectations=False)

    checkpoint = SimpleCheckpoint(
        name="task_custom_diff_checkpoint",
        data_context=context,
        validations=[
            {
                "batch_request": validator.active_batch.batch_request,
                "expectation_suite_name": validator.expectation_suite_name,
            }
        ],
    )
    cp_result = checkpoint.run(result_format=RUNTIME_RESULT_FORMAT)
    print("Checkpoint run success:", cp_result.get("success"))

    context.build_data_docs()
    try:
        context.open_data_docs()
    except Exception as ex:
        print("Data Docs could not be opened automatically:", ex)
        print("Open from:", Path(context.root_directory) / "data_docs")


if __name__ == "__main__":
    main()
