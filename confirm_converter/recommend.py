from __future__ import annotations

from typing import Dict, List

import pandas as pd


def _cardinality(series: pd.Series) -> int:
	return int(series.nunique(dropna=True))


def summarize_field(df: pd.DataFrame, column: str) -> pd.DataFrame:
	series = df[column]
	stats = {
		"column": column,
		"non_null": int(series.notna().sum()),
		"nulls": int(series.isna().sum()),
		"unique": int(series.nunique(dropna=True)),
	}
	if pd.api.types.is_numeric_dtype(series):
		stats.update(
			{"min": float(series.min(skipna=True)), "max": float(series.max(skipna=True))}
		)
	return pd.DataFrame([stats])


def recommend_variables(df: pd.DataFrame) -> pd.DataFrame:
	recs: List[dict] = []
	for col in df.columns:
		series = df[col]
		card = _cardinality(series)
		is_cand = card <= 50 or pd.api.types.is_categorical_dtype(series)
		recs.append(
			{
				"column": col,
				"unique": card,
				"dtype": str(series.dtype),
				"recommended": bool(is_cand),
			}
		)
	frame = pd.DataFrame(recs).sort_values(["recommended", "unique"], ascending=[False, True])
	return frame

