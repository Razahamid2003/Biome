"""Great Expectations 0.18.21

Custom expectation and diagnostic renderer:

- Metric: pass rows where |A - B| is within [min_threshold, max_threshold] (inclusive).
- Expectation: validates inputs and augments EVR with details for rendering.
- Renderer (diagnostic.unexpected_table): shows ONLY failed rows, with red text via inline HTML.
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


# ---------------- Metric Provider ---------------- #
class ColumnPairValuesDiffWithinRange(ColumnPairMapMetricProvider):
    """True iff abs(A - B) is within [min_threshold, max_threshold] inclusive."""

    condition_metric_name = "column_pair_values.diff_within_range"
    condition_value_keys = ("min_threshold", "max_threshold", "sort", "sort_key", "ignore_row_if")

    @column_pair_condition_partial(engine=PandasExecutionEngine)
    def _pandas(  # type: ignore[override]
        cls,
        column_A: pd.Series,
        column_B: pd.Series,
        **kwargs: Any,
    ) -> pd.Series:
        """Pandas engine implementation for the condition.

        Returns a boolean Series: True means the row satisfies the range (or is ignored).
        """
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


# ---------------- Expectation ---------------- #
class ExpectColumnPairValuesDiffWithinRange(ColumnPairMapExpectation):
    """Expect |column_A - column_B| within an inclusive range.

    Inputs:
        - column_A (str), column_B (str)
        - min_threshold (float | None), max_threshold (float | None)
        - sort (bool), sort_key (str | None)
        - ignore_row_if (str): "both_values_are_missing" | "either_value_is_missing" | "neither"
        - mostly (float)
    """

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
    library_metadata = {
        "tags": ["custom", "column_pair", "diff", "range"],
        "contributors": ["@hatchi.08251973"],
    }

    # Minimal gallery examples (kept compact for CI/lint)
    examples: List[Dict[str, Any]] = [
        {
            "data": {
                "col_a": [10, 12, 15, 20, 13, 9],
                "col_b": [8, 10, 16, 19, 9, 13],
            },
            "tests": [
                {
                    "title": "pass_all_between_1_and_4",
                    "exact_match_out": False,
                    "include_in_gallery": True,
                    "in": {
                        "column_A": "col_a",
                        "column_B": "col_b",
                        "min_threshold": 1,
                        "max_threshold": 4,
                        "mostly": 1.0,
                    },
                    "out": {"success": True},
                },
                {
                    "title": "fail_when_stricter_max_2",
                    "exact_match_out": False,
                    "include_in_gallery": True,
                    "in": {
                        "column_A": "col_a",
                        "column_B": "col_b",
                        "min_threshold": 1,
                        "max_threshold": 2,
                        "mostly": 1.0,
                    },
                    "out": {"success": False},
                },
            ],
        }
    ]

    # --- Input validation ---
    def validate_configuration(self, configuration: Optional[dict] = None) -> bool:
        """Validate user-supplied kwargs for the expectation."""
        super().validate_configuration(configuration)
        config = configuration or self.configuration
        if config is None:
            raise InvalidExpectationConfigurationError("No configuration provided.")
        kwargs: Dict[str, Any] = dict(config.kwargs or {})

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
                raise InvalidExpectationConfigurationError("sort_key must equal column_A or column_B.")

        ignore_row_if = kwargs.get("ignore_row_if", "both_values_are_missing")
        allowed = {"both_values_are_missing", "either_value_is_missing", "neither"}
        if ignore_row_if not in allowed:
            raise InvalidExpectationConfigurationError(
                f"ignore_row_if must be one of {sorted(allowed)}."
            )

        return True

    # ---- Enrich EVR with ALL rows (ad-hoc validator path) ----
    def validate(  # type: ignore[override]
        self,
        validator: Any,
        configuration: Optional[dict] = None,
        runtime_configuration: Optional[dict] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run base validation then attach details['all_rows'] with pass/fail for rendering."""
        evr: Dict[str, Any] = super().validate(
            validator,
            configuration=configuration,
            runtime_configuration=runtime_configuration,
            **kwargs,
        )
        try:
            cfg = configuration or self.configuration
            assert cfg is not None
            params = dict(cfg.kwargs or {})

            col_A: str = params["column_A"]
            col_B: str = params["column_B"]
            min_th = params.get("min_threshold")
            max_th = params.get("max_threshold")
            sort_flag = bool(params.get("sort", False))
            sort_key = params.get("sort_key")
            ignore_row_if = params.get("ignore_row_if", "both_values_are_missing")

            df_all: pd.DataFrame = validator.active_batch.data.dataframe  # type: ignore[assignment]
            df = df_all[[col_A, col_B]].copy()
            if sort_flag:
                df = df.sort_values(by=sort_key)

            diff = (df[col_A] - df[col_B]).abs()
            passed = pd.Series(True, index=df.index, dtype=bool)
            if min_th is not None:
                passed &= diff >= float(min_th)
            if max_th is not None:
                passed &= diff <= float(max_th)

            if ignore_row_if == "both_values_are_missing":
                passed |= df[col_A].isna() & df[col_B].isna()
            elif ignore_row_if == "either_value_is_missing":
                passed |= df[col_A].isna() | df[col_B].isna()
            elif ignore_row_if == "neither":
                passed &= df[col_A].notna() & df[col_B].notna()

            if sort_flag:
                df = df.sort_index()
                passed = passed.sort_index()

            rows: List[Dict[str, Any]] = []
            for i in df.index:
                rows.append(
                    {
                        "index": int(i),
                        col_A: None if pd.isna(df.loc[i, col_A]) else df.loc[i, col_A],
                        col_B: None if pd.isna(df.loc[i, col_B]) else df.loc[i, col_B],
                        "passed": bool(passed.loc[i]),
                    }
                )

            evr_result = evr.get("result", {}) or {}
            details = evr_result.get("details", {}) or {}
            details["all_rows"] = {"columns": [col_A, col_B], "rows": rows}
            evr_result["details"] = details
            evr["result"] = evr_result
        except Exception:
            # Do not fail validation if rendering enrichment fails.
            pass
        return evr

    # ---- Enrich EVR with ALL rows (checkpoint/graph path) ----
    def _validate(  # type: ignore[override]
        self,
        configuration: dict,
        metrics: Dict[str, Any],
        runtime_configuration: Optional[dict] = None,
        execution_engine: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Attach details['all_rows'] during graph validation for Data Docs rendering."""
        res: Dict[str, Any] = super()._validate(
            configuration,
            metrics,
            runtime_configuration,
            execution_engine,
        )
        try:
            params = dict(configuration.get("kwargs") or {})
            col_A: str = params["column_A"]
            col_B: str = params["column_B"]
            min_th = params.get("min_threshold")
            max_th = params.get("max_threshold")
            sort_flag = bool(params.get("sort", False))
            sort_key = params.get("sort_key")
            ignore_row_if = params.get("ignore_row_if", "both_values_are_missing")

            if not isinstance(execution_engine, PandasExecutionEngine):
                return res

            df: pd.DataFrame = execution_engine.get_domain_records(  # type: ignore[assignment]
                domain_kwargs={"column_A": col_A, "column_B": col_B}
            )
            df = df[[col_A, col_B]].copy()
            if sort_flag:
                df = df.sort_values(by=sort_key)

            diff = (df[col_A] - df[col_B]).abs()
            passed = pd.Series(True, index=df.index, dtype=bool)
            if min_th is not None:
                passed &= diff >= float(min_th)
            if max_th is not None:
                passed &= diff <= float(max_th)

            if ignore_row_if == "both_values_are_missing":
                passed |= df[col_A].isna() & df[col_B].isna()
            elif ignore_row_if == "either_value_is_missing":
                passed |= df[col_A].isna() | df[col_B].isna()
            elif ignore_row_if == "neither":
                passed &= df[col_A].notna() & df[col_B].notna()

            if sort_flag:
                df = df.sort_index()
                passed = passed.sort_index()

            rows: List[Dict[str, Any]] = []
            for i in df.index:
                rows.append(
                    {
                        "index": int(i),
                        col_A: None if pd.isna(df.loc[i, col_A]) else df.loc[i, col_A],
                        col_B: None if pd.isna(df.loc[i, col_B]) else df.loc[i, col_B],
                        "passed": bool(passed.loc[i]),
                    }
                )

            res.setdefault("result", {})
            res["result"].setdefault("details", {})
            res["result"]["details"]["all_rows"] = {"columns": [col_A, col_B], "rows": rows}
        except Exception:
            # Do not fail validation if rendering enrichment fails.
            pass
        return res

    # --- Prescriptive text ---
    @classmethod
    @renderer(renderer_type="renderer.prescriptive")
    def _prescriptive_renderer(  # type: ignore[override]
        cls,
        configuration: Optional[dict] = None,
        result: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[RenderedStringTemplateContent]:
        """Human-readable statement describing the expectation."""
        cfg = configuration or (result and result.get("expectation_config")) or {}
        params = (cfg.get("kwargs") or {})
        col_a = params.get("column_A", "column_A")
        col_b = params.get("column_B", "column_B")
        min_t = params.get("min_threshold")
        max_t = params.get("max_threshold")

        parts: List[str] = []
        if min_t is not None:
            parts.append(f"≥ {min_t}")
        if max_t is not None:
            parts.append(f"≤ {max_t}")
        rng = " and ".join(parts) if parts else "any value"

        return [
            RenderedStringTemplateContent(
                string_template={"template": f"|{col_a} − {col_b}| within {rng}."}
            )
        ]

    # --- Diagnostic table: ONLY failures; red text via inline HTML (no background) ---
    @classmethod
    @renderer(renderer_type="renderer.diagnostic.unexpected_table")
    def _unexpected_table_renderer(  # type: ignore[override]
        cls,
        result: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[RenderedTableContent] | List[RenderedStringTemplateContent]:
        """Render a compact table of only failing rows with inline red text styling."""
        if not result:
            return []

        res = result.get("result") or {}
        unexpected = res.get("unexpected_list") or res.get("partial_unexpected_list") or []
        idx_list = res.get("unexpected_index_list")  # may be None if not COMPLETE format

        cfg = result.get("expectation_config") or {}
        params = (cfg.get("kwargs") or {})
        col_a = params.get("column_A", "column_A")
        col_b = params.get("column_B", "column_B")

        if not unexpected:
            return [RenderedStringTemplateContent(string_template={"template": "No unexpected rows."})]

        def fail_html(value: Any) -> str:
            """Inline HTML for red text only; avoids backgrounds/scrollbars in Data Docs."""
            txt = "—" if value is None else str(value)
            return (
                '<span style="color:#B3261E; font-weight:600; '
                'background:none; padding:0; border:none; border-radius:0; '
                'white-space:nowrap; overflow:visible; display:inline; box-shadow:none;">'
                f"{txt}</span>"
            )

        header_row = ["index", col_a, col_b] if idx_list is not None else [col_a, col_b]
        table: List[List[RenderedStringTemplateContent]] = []

        for i, pair in enumerate(unexpected):
            a = pair[0] if isinstance(pair, (list, tuple)) and len(pair) > 0 else None
            b = pair[1] if isinstance(pair, (list, tuple)) and len(pair) > 1 else None

            row_cells: List[RenderedStringTemplateContent] = []
            if idx_list is not None:
                row_cells.append(RenderedStringTemplateContent(string_template={"template": fail_html(idx_list[i])}))
            row_cells.append(RenderedStringTemplateContent(string_template={"template": fail_html(a)}))
            row_cells.append(RenderedStringTemplateContent(string_template={"template": fail_html(b)}))
            table.append(row_cells)

        header = RenderedStringTemplateContent(
            string_template={"template": f"Failed rows for {col_a} vs {col_b} (only unexpected shown)"}
        )

        return [
            RenderedTableContent(
                header=header,
                header_row=header_row,
                table=table,
                table_options={"search": True, "icon-size": "sm"},
            )
        ]


if __name__ == "__main__":
    # Quick smoke check (prints the checklist; does not execute validation).
    ExpectColumnPairValuesDiffWithinRange().print_diagnostic_checklist()
