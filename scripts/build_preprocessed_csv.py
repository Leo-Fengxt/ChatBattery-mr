#!/usr/bin/env python3
"""
Build ChatBattery retrieval DB files (data/*_battery/preprocessed.csv) from a
raw list/export of chemical formulas (e.g., from ICSD).

The downstream code expects a CSV with columns:
  - formula: string chemical formula
  - capacity: theoretical capacity computed by Domain_Agent

Notes:
- This script does NOT download from ICSD (licensing/access varies). Instead,
  you provide an export file containing formulas.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import pandas as pd

from ChatBattery.domain_agent import Domain_Agent, parse_formula


DEFAULT_LI_OUTPUT = os.path.join("data", "Li_battery", "preprocessed.csv")
DEFAULT_NA_OUTPUT = os.path.join("data", "Na_battery", "preprocessed.csv")


def _infer_formula_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "formula",
        "Formula",
        "chemical_formula",
        "Chemical formula",
        "chemical formula",
        "composition",
        "Composition",
        "compound",
        "Compound",
        "reduced_formula",
        "pretty_formula",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _read_formulas(input_file: str, formula_column: Optional[str]) -> Tuple[List[str], str]:
    ext = os.path.splitext(input_file)[1].lower()

    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(input_file, sep=sep)

        if formula_column is None:
            formula_column = _infer_formula_column(df)
        if formula_column is None or formula_column not in df.columns:
            raise ValueError(
                f"Could not infer formula column from {list(df.columns)}. "
                "Pass --formula_column explicitly."
            )

        formulas = df[formula_column].astype(str).tolist()
        return formulas, f"{os.path.basename(input_file)}[{formula_column}]"

    # Fallback: treat as newline-delimited formulas
    formulas: List[str] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            formulas.append(line)
    return formulas, os.path.basename(input_file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, type=str, help="ICSD export (csv/tsv) or newline-delimited text")
    parser.add_argument(
        "--task_index",
        type=int,
        default=101,
        choices=[101, 102],
        help="101 for Li DB, 102 for Na DB",
    )
    parser.add_argument(
        "--formula_column",
        type=str,
        default=None,
        help="Column name in the input CSV/TSV that contains formulas (optional if inferable)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output CSV path. Defaults to data/Li_battery/preprocessed.csv or data/Na_battery/preprocessed.csv",
    )
    parser.add_argument(
        "--deduplicate_exact",
        action="store_true",
        help="Drop exact-duplicate formula strings before writing (keeps first occurrence)",
    )
    parser.add_argument(
        "--errors_file",
        type=str,
        default=None,
        help="Optional CSV to write formulas that could not be processed, with error reasons",
    )
    args = parser.parse_args()

    target_element = "Li" if args.task_index == 101 else "Na"
    output_file = args.output_file or (DEFAULT_LI_OUTPUT if args.task_index == 101 else DEFAULT_NA_OUTPUT)

    formulas, source_desc = _read_formulas(args.input_file, args.formula_column)

    records = []
    errors = []

    for raw in formulas:
        formula = str(raw).strip().replace(" ", "")
        if not formula:
            continue

        # Filter to formulas containing the target element (Li / Na)
        try:
            comp = parse_formula(formula)
        except Exception as e:
            errors.append({"formula": formula, "stage": "parse_formula", "error": repr(e)})
            continue

        if target_element not in comp or comp[target_element] == 0:
            continue

        try:
            capacity = Domain_Agent.calculate_theoretical_capacity(formula, args.task_index)
        except Exception as e:
            errors.append({"formula": formula, "stage": "capacity", "error": repr(e)})
            continue

        records.append({"formula": formula, "capacity": capacity})

    out_df = pd.DataFrame(records)
    if args.deduplicate_exact and not out_df.empty:
        out_df = out_df.drop_duplicates(subset=["formula"], keep="first")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    out_df.to_csv(output_file, index=False)

    print(f"[ChatBattery] Built retrieval DB for task_index={args.task_index} ({target_element})")
    print(f"[ChatBattery] Source: {source_desc}")
    print(f"[ChatBattery] Input formulas: {len(formulas)}")
    print(f"[ChatBattery] Output rows: {len(out_df)} -> {output_file}")
    if errors:
        print(f"[ChatBattery] Skipped rows with errors: {len(errors)}")

    if args.errors_file is not None:
        err_df = pd.DataFrame(errors)
        os.makedirs(os.path.dirname(args.errors_file) or ".", exist_ok=True)
        err_df.to_csv(args.errors_file, index=False)
        print(f"[ChatBattery] Wrote errors -> {args.errors_file}")


if __name__ == "__main__":
    main()

