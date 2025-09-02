from __future__ import annotations

import io
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text


def load_tabular_file(file_obj) -> pd.DataFrame:
	name = getattr(file_obj, "name", "uploaded")
	lower = name.lower()
	if lower.endswith(".csv"):
		return pd.read_csv(file_obj)
	if lower.endswith((".xlsx", ".xls")):
		return pd.read_excel(file_obj)
	raised = ValueError(f"Unsupported tabular file: {name}")
	raise raised


def load_json_records(file_obj) -> pd.DataFrame:
	# Accept JSON array of objects or JSON Lines
	data = file_obj.read()
	if isinstance(data, bytes):
		data = data.decode("utf-8")
	data = data.strip()
	try:
		if data.startswith("["):
			return pd.read_json(io.StringIO(data))
		# JSON Lines fallback
		return pd.read_json(io.StringIO(data), lines=True)
	except ValueError as exc:
		raise ValueError(f"Invalid JSON records: {exc}")


def load_sql_query(connection_url: str, query: str, table_fallback: str = "") -> pd.DataFrame:
	engine = create_engine(connection_url)
	with engine.begin() as conn:
		if query:
			return pd.read_sql_query(text(query), conn)
		if not table_fallback:
			raise ValueError("Provide a query or a table name.")
		return pd.read_sql_table(table_fallback, conn)

