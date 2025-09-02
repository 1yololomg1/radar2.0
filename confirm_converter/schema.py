from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def _infer_type(series: pd.Series) -> str:
	# Basic inference for categorical vs numeric vs datetime
	if pd.api.types.is_categorical_dtype(series):
		return "categorical"
	if pd.api.types.is_bool_dtype(series):
		return "categorical"
	if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
		return "numeric"
	if pd.api.types.is_datetime64_any_dtype(series):
		return "datetime"
	# Try parseable datetime
	try:
		pd.to_datetime(series.dropna().head(100), errors="raise")
		return "datetime"
	except Exception:
		pass
	# Fallback to categorical for low unique count else text
	unique = series.dropna().nunique()
	return "categorical" if unique <= 50 else "text"


def detect_schema(df: pd.DataFrame) -> pd.DataFrame:
	records = []
	for col in df.columns:
		series = df[col]
		inferred = _infer_type(series)
		records.append(
			{
				"column": col,
				"inferred_type": inferred,
				"non_null": int(series.notna().sum()),
				"nulls": int(series.isna().sum()),
				"unique": int(series.nunique(dropna=True)),
				"min": series.min() if inferred in {"numeric", "datetime"} else None,
				"max": series.max() if inferred in {"numeric", "datetime"} else None,
			}
		)
	return pd.DataFrame.from_records(records)


def ensure_categorical(
	series: pd.Series,
	method: str = "quantile",
	n_bins: int = 4,
	custom_edges: Optional[str] = None,
) -> pd.Series:
	if pd.api.types.is_categorical_dtype(series) or pd.api.types.is_bool_dtype(series):
		return series.astype("category")
	if pd.api.types.is_datetime64_any_dtype(series):
		# Convert to ordinal (days) for binning, then label back with intervals
		ordinal = series.view("int64")
		return ensure_categorical(pd.Series(ordinal, index=series.index), method, n_bins, custom_edges)
	if not (pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series)):
		return series.astype("category")

	values = series.astype(float)
	if method == "quantile":
		binned = pd.qcut(values, q=n_bins, duplicates="drop")
		return binned.astype("category")
	if method == "width":
		binned = pd.cut(values, bins=n_bins)
		return binned.astype("category")
	if method == "custom":
		if not custom_edges:
			raise ValueError("Custom edges required for custom binning")
		edges = [float(x.strip()) for x in custom_edges.split(",") if x.strip()]
		if len(edges) < 2:
			raise ValueError("Provide at least two edges")
		binned = pd.cut(values, bins=edges, include_lowest=True)
		return binned.astype("category")

	raise ValueError(f"Unknown method: {method}")

