"""Great Expectations 0.18.21 — minimal custom expectation.

- Validates: |A - B| within [min_threshold, max_threshold] (inclusive).
- Prescriptive renderer: natural-language sentence from kwargs.
- Diagnostic renderer: only failed rows, red cells, centered headers,
  wider index, no extra header text.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from great_expectations.exceptions.exceptions import InvalidExpectationConfigurationError
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.expectations.expectation import ColumnPairMapExpectation
from great_expectations.expectations.metrics.map_metric_provider import (
    ColumnPairMapMetricProvider,
    column_pair_condition_partial,
)
from great_expectations.render import RenderedStringTemplateContent, RenderedTableContent
from great_expectations.render.renderer.renderer import renderer


# ---------- Metric Provider ----------
class ColumnPairValuesDiffWithinRange(ColumnPairMapMetricProvider):
    """True iff abs(A - B) within [min_threshold, max_threshold] (inclusive)."""

    condition_metric_name = "column_pair_values.diff_within_range"
    condition_value_keys = (
        "min_threshold",
        "max_threshold",
        "sort",
        "sort_key",
        "ignore_row_if",
    )

    @column_pair_condition_partial(engine=PandasExecutionEngine)
    def _pandas(
        cls,
        column_A: pd.Series,
        column_B: pd.Series,
        **kwargs: Any,
    ) -> pd.Series:
        min_threshold = kwargs.get("min_threshold")
        max_threshold = kwargs.get("max_threshold")
        sort = bool(kwargs.get("sort", False))
        sort_key = kwargs.get("sort_key")
        ignore_row_if = kwargs.get("ignore_row_if", "both_values_are_missing")

        name_A = getattr(column_A, "name", "column_A")
        name_B = getattr(column_B, "name", "column_B")
        df = pd.DataFrame({name_A: column_A, name_B: column_B})

        if sort:
            if not isinstance(sort_key, str) or sort_key not in (name_A, name_B):
                raise InvalidExpectationConfigurationError(
                    "When sort=True, sort_key must equal column_A "
                    f"('{name_A}') or column_B ('{name_B}'). "
                    f"Got: {sort_key!r}"
                )
            df = df.sort_values(by=sort_key)

        diff = (df[name_A] - df[name_B]).abs()
        mask = pd.Series(True, index=df.index, dtype=bool)
        if min_threshold is not None:
            mask &= diff >= float(min_threshold)
        if max_threshold is not None:
            mask &= diff <= float(max_threshold)

        if ignore_row_if == "both_values_are_missing":
            mask |= df[name_A].isna() & df[name_B].isna()
        elif ignore_row_if == "either_value_is_missing":
            mask |= df[name_A].isna() | df[name_B].isna()
        elif ignore_row_if == "neither":
            mask &= df[name_A].notna() & df[name_B].notna()

        if sort:
            mask = mask.sort_index()

        return mask


# ---------- Expectation ----------
class ExpectColumnPairValuesDiffWithinRange(ColumnPairMapExpectation):
    """Expect |column_A - column_B| within an inclusive range."""

    expectation_type = "expect_column_pair_values_diff_within_range"
    map_metric = "column_pair_values.diff_within_range"

    success_keys = (
        "column_A",
        "column_B",
        "min_threshold",
        "max_threshold",
        "mostly",
        "sort",
        "sort_key",
        "ignore_row_if",
    )
    default_kwarg_values: Dict[str, Any] = {
        "min_threshold": None,
        "max_threshold": None,
        "mostly": 1.0,
        "sort": False,
        "sort_key": None,
        "ignore_row_if": "both_values_are_missing",
    }

    # ---- Input validation ----
    def validate_configuration(self, configuration: Optional[dict] = None) -> bool:
        super().validate_configuration(configuration)
        config = configuration or self.configuration
        if config is None:
            raise InvalidExpectationConfigurationError("No configuration provided.")

        if hasattr(config, "kwargs"):
            kwargs: Dict[str, Any] = dict(getattr(config, "kwargs", {}) or {})
        elif isinstance(config, dict):
            kwargs = dict(config.get("kwargs") or {})
        else:
            kwargs = {}

        for key in ("column_A", "column_B"):
            if key not in kwargs or not isinstance(kwargs[key], str) or not kwargs[key]:
                raise InvalidExpectationConfigurationError(f"{key} must be a non-empty string.")

        min_th = kwargs.get("min_threshold")
        max_th = kwargs.get("max_threshold")
        if min_th is None and max_th is None:
            raise InvalidExpectationConfigurationError(
                "Provide at least one of min_threshold or max_threshold."
            )
        if min_th is not None and not isinstance(min_th, (int, float)):
            raise InvalidExpectationConfigurationError("min_threshold must be numeric.")
        if max_th is not None and not isinstance(max_th, (int, float)):
            raise InvalidExpectationConfigurationError("max_threshold must be numeric.")
        if min_th is not None and max_th is not None and float(min_th) > float(max_th):
            raise InvalidExpectationConfigurationError(
                "min_threshold cannot be greater than max_threshold."
            )

        sort = kwargs.get("sort", False)
        if sort is not None and not isinstance(sort, bool):
            raise InvalidExpectationConfigurationError("sort must be a boolean.")
        sort_key = kwargs.get("sort_key")
        if sort:
            if not isinstance(sort_key, str) or not sort_key:
                raise InvalidExpectationConfigurationError(
                    "When sort=True, sort_key must be a non-empty string."
                )
            if sort_key not in (kwargs["column_A"], kwargs["column_B"]):
                raise InvalidExpectationConfigurationError(
                    "sort_key must equal column_A or column_B."
                )

        ignore_row_if = kwargs.get("ignore_row_if", "both_values_are_missing")
        allowed = {"both_values_are_missing", "either_value_is_missing", "neither"}
        if ignore_row_if not in allowed:
            raise InvalidExpectationConfigurationError(
                f"ignore_row_if must be one of {sorted(allowed)}."
            )

        return True

    # ---- Prescriptive text ----
    @classmethod
    @renderer(renderer_type="renderer.prescriptive")
    def _prescriptive_renderer(
        cls,
        configuration: Optional[dict] = None,
        result: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[RenderedStringTemplateContent]:
        cfg = configuration or (result and result.get("expectation_config")) or {}

        if hasattr(cfg, "kwargs"):
            params: Dict[str, Any] = dict(getattr(cfg, "kwargs", {}) or {})
        elif isinstance(cfg, dict):
            params = dict(cfg.get("kwargs") or {})
        else:
            params = {}

        col_a = params.get("column_A", "column_A")
        col_b = params.get("column_B", "column_B")
        min_t = params.get("min_threshold")
        max_t = params.get("max_threshold")

        def fmt_num(x: Any) -> str:
            try:
                xv = float(x)
                return str(int(xv)) if xv.is_integer() else str(xv)
            except Exception:
                return str(x)

        if max_t is not None and min_t is not None:
            if float(min_t) == float(max_t):
                text = (
                    "The difference between values of "
                    f"{col_a} and {col_b} must be equal to {fmt_num(min_t)}."
                )
            else:
                text = (
                    "The difference between values of "
                    f"{col_a} and {col_b} must be less than {fmt_num(max_t)} "
                    f"and greater than {fmt_num(min_t)}."
                )
        elif max_t is not None:
            text = (
                "The difference between values of "
                f"{col_a} and {col_b} must be less than {fmt_num(max_t)}."
            )
        elif min_t is not None:
            text = (
                "The difference between values of "
                f"{col_a} and {col_b} must be greater than {fmt_num(min_t)}."
            )
        else:
            text = (
                "The difference between values of " f"{col_a} and {col_b} must be within a range."
            )

        return [RenderedStringTemplateContent(string_template={"template": text})]

    # ---- Diagnostic table ----
    @classmethod
    @renderer(renderer_type="renderer.diagnostic.unexpected_table")
    def _unexpected_table_renderer(
        cls,
        result: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[RenderedTableContent] | List[RenderedStringTemplateContent]:
        if not result:
            return []

        res = result.get("result") or {}
        unexpected = res.get("unexpected_list") or res.get("partial_unexpected_list") or []
        idx_list = res.get("unexpected_index_list")

        cfg = result.get("expectation_config") or {}
        if hasattr(cfg, "kwargs"):
            params: Dict[str, Any] = dict(getattr(cfg, "kwargs", {}) or {})
        elif isinstance(cfg, dict):
            params = dict(cfg.get("kwargs") or {})
        else:
            params = {}
        col_a = params.get("column_A", "column_A")
        col_b = params.get("column_B", "column_B")

        if not unexpected:
            return [
                RenderedStringTemplateContent(string_template={"template": "No unexpected rows."})
            ]

        def fail_html(value: Any) -> str:
            txt = "—" if value is None else str(value)
            return (
                "<span style='color:#B3261E; font-weight:600; "
                "background:none; padding:0; border:none; border-radius:0; "
                "white-space:nowrap; overflow:visible; display:inline; "
                f"box-shadow:none;'>{txt}</span>"
            )

        def index_html(value: Any) -> str:
            txt = "—" if value is None else str(value)
            return (
                "<span style='color:#B3261E; font-weight:600; "
                "display:inline-block; min-width:56px; text-align:center; "
                "background:none; padding:0; border:none; border-radius:0; "
                f"white-space:nowrap; overflow:visible; box-shadow:none;'>{txt}</span>"
            )

        def center_header_html(text: str) -> str:
            return f"<div style='text-align:center;'>{text}</div>"

        header_row: List[RenderedStringTemplateContent | str] = []
        if idx_list is not None:
            header_row.append(
                RenderedStringTemplateContent(
                    string_template={
                        "template": (
                            "<span style='display:inline-block; min-width:56px; "
                            "text-align:center;'>index</span>"
                        )
                    }
                )
            )
        header_row.append(
            RenderedStringTemplateContent(string_template={"template": center_header_html(col_a)})
        )
        header_row.append(
            RenderedStringTemplateContent(string_template={"template": center_header_html(col_b)})
        )

        table_rows: List[List[RenderedStringTemplateContent]] = []
        for i, pair in enumerate(unexpected):
            a = pair[0] if isinstance(pair, (list, tuple)) and len(pair) > 0 else None
            b = pair[1] if isinstance(pair, (list, tuple)) and len(pair) > 1 else None

            row_cells: List[RenderedStringTemplateContent] = []
            if idx_list is not None:
                row_cells.append(
                    RenderedStringTemplateContent(
                        string_template={"template": index_html(idx_list[i])}
                    )
                )
            row_cells.append(
                RenderedStringTemplateContent(string_template={"template": fail_html(a)})
            )
            row_cells.append(
                RenderedStringTemplateContent(string_template={"template": fail_html(b)})
            )
            table_rows.append(row_cells)

        return [
            RenderedTableContent(
                header=None,
                header_row=header_row,
                table=table_rows,
                table_options={"search": True, "icon-size": "sm"},
            )
        ]
