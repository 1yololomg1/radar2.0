from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def build_contingency_table(
	df: pd.DataFrame,
	row_field: str,
	col_field: str,
	normalize: bool = False,
	dropna_levels: bool = True,
	min_count: int = 0,
) -> pd.DataFrame:
	data = df[[row_field, col_field]].copy()
	if dropna_levels:
		data = data.dropna(subset=[row_field, col_field])

	table = pd.crosstab(
		index=data[row_field],
		columns=data[col_field],
		dropna=dropna_levels,
		normalize="all" if normalize else False,
	)
	if not normalize and min_count > 0:
		table = table.where(table >= min_count)

	return table.fillna(0 if not normalize else 0.0)


def validate_contingency_table(table: pd.DataFrame) -> Tuple[bool, List[str]]:
	notes: List[str] = []
	valid = True

	if table.empty:
		return False, ["Table is empty"]

	# Check for sparse rows/columns
	row_sums = table.sum(axis=1)
	col_sums = table.sum(axis=0)
	if (row_sums == 0).any():
		valid = False
		notes.append("At least one row has zero total count")
	if (col_sums == 0).any():
		valid = False
		notes.append("At least one column has zero total count")

	# Small expected counts heuristic (only for counts, not proportions)
	if table.values.dtype.kind in {"i", "u", "f"} and (table.values.max() > 1.0):
		total = table.values.sum()
		expected = table.sum(axis=1).values.reshape(-1, 1) * table.sum(axis=0).values.reshape(1, -1) / float(total) if total > 0 else np.zeros_like(table.values)
		if (expected < 5).sum() > 0:
			notes.append("Some cells have expected count < 5; chi-square assumptions may be violated")

	return valid, notes

