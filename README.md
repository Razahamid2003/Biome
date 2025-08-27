# GX 0.18.21 — Custom Column-Pair Diff Expectation

This Task implements a custom Great Expectations expectation that
checks whether the absolute difference between two columns is within a threshold
range and renders failed rows in Data Docs.

## Contents

- **Custom expectation**
  `gx/plugins/expectations/expect_column_pair_values_diff_within_range.py`
- **Runner script**
  `runner.py` — loads CSV, reads config JSON, creates a suite, runs a checkpoint, builds or opens Data Docs
- **Config**
  `expectation_config.json` — parameters for the expectation and suite names

## Installation and Deployment
python -m pip install -r requirements.txt
python runner.py
