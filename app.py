import streamlit as st
import pandas as pd
from typing import Optional, Tuple

from confirm_converter.data_import import (
	load_tabular_file,
	load_json_records,
	load_sql_query,
)
from confirm_converter.schema import detect_schema, ensure_categorical
from confirm_converter.recommend import recommend_variables, summarize_field
from confirm_converter.contingency import (
	build_contingency_table,
	validate_contingency_table,
)


st.set_page_config(page_title="CONFIRM Data Converter", layout="wide")


def init_state() -> None:
	if "source_df" not in st.session_state:
		st.session_state["source_df"] = None
	if "working_df" not in st.session_state:
		st.session_state["working_df"] = None
	if "schema" not in st.session_state:
		st.session_state["schema"] = None
	if "row_var" not in st.session_state:
		st.session_state["row_var"] = None
	if "col_var" not in st.session_state:
		st.session_state["col_var"] = None
	if "binnings" not in st.session_state:
		st.session_state["binnings"] = {}


def sidebar_import() -> Optional[pd.DataFrame]:
	st.sidebar.header("1) Import Data")
	file = st.sidebar.file_uploader(
		"Upload CSV / Excel (.xlsx) / JSON (records)",
		type=["csv", "xlsx", "xls", "json"],
	)

	uploaded_df: Optional[pd.DataFrame] = None
	if file is not None:
		if file.name.lower().endswith((".csv", ".xls", ".xlsx")):
			try:
				uploaded_df = load_tabular_file(file)
			except Exception as exc:
				st.sidebar.error(f"File load error: {exc}")
		elif file.name.lower().endswith(".json"):
			try:
				uploaded_df = load_json_records(file)
			except Exception as exc:
				st.sidebar.error(f"JSON load error: {exc}")

	st.sidebar.markdown("---")
	with st.sidebar.expander("Or connect to a SQL database"):
		connection_url = st.text_input(
			"SQLAlchemy connection URL",
			help=(
				"Examples: sqlite:///my.db, postgresql+psycopg2://user:pass@host/db, "
				"mysql+pymysql://user:pass@host/db"
			),
		)
		query = st.text_area(
			"Custom SQL query (optional)",
			height=110,
			help="If empty, a default SELECT * FROM <table> is required via table name below.",
		)
		table_fallback = st.text_input(
			"Fallback table name if no query provided",
			value="",
			help="Only used when query above is empty.",
		)
		if st.button("Run SQL import", use_container_width=True):
			if not connection_url:
				st.sidebar.error("Provide a connection URL.")
			else:
				try:
					uploaded_df = load_sql_query(connection_url, query.strip(), table_fallback.strip())
				except Exception as exc:
					st.sidebar.error(f"SQL load error: {exc}")

	return uploaded_df


def schema_and_preview(df: pd.DataFrame) -> None:
	st.subheader("Data Preview")
	st.caption("Showing up to first 100 rows")
	st.dataframe(df.head(100), use_container_width=True)

	st.markdown("---")
	st.subheader("Detected Schema & Field Statistics")
	schema = detect_schema(df)
	st.session_state["schema"] = schema
	st.dataframe(schema, use_container_width=True)


def variable_selection_ui(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
	st.markdown("---")
	st.header("2) Variable Selection")

	left, right = st.columns(2)

	with left:
		st.subheader("Row Variable (x-axis)")
		row_var = st.selectbox("Select row variable", options=df.columns.tolist(), index=None)
		if row_var:
			row_stats = summarize_field(df, row_var)
			st.write(row_stats)

	with right:
		st.subheader("Column Variable (y-axis)")
		col_var = st.selectbox("Select column variable", options=df.columns.tolist(), index=None)
		if col_var:
			col_stats = summarize_field(df, col_var)
			st.write(col_stats)

	recs = recommend_variables(df)
	if not row_var and not col_var and len(recs) > 0:
		st.info("Recommended variables based on cardinality and data type")
		st.dataframe(recs, use_container_width=True)

	return row_var, col_var


def maybe_categorize(df: pd.DataFrame, var: str, key_prefix: str) -> Tuple[pd.DataFrame, str]:
	field_info = st.session_state["schema"].set_index("column").loc[var]
	is_continuous = field_info["inferred_type"] in {"numeric", "datetime"}

	if not is_continuous:
		return df, var

	st.warning(f"{var} appears continuous and may require categorization for contingency tables.")
	method = st.selectbox(
		f"Binning method for {var}",
		["Quantile", "Equal width", "Custom edges"],
		key=f"{key_prefix}_method",
	)
	bins = st.number_input(
		f"Number of bins for {var}", min_value=2, max_value=20, value=4, step=1, key=f"{key_prefix}_bins"
	)
	custom_edges_text = ""
	if method == "Custom edges":
		custom_edges_text = st.text_input(
			f"Comma-separated edges for {var}",
			value="",
			help="Example: 0,10,20,50,100",
			key=f"{key_prefix}_edges",
		)

	binned_col = f"{var}__binned"
	if st.button(f"Apply categorization for {var}", key=f"{key_prefix}_apply"):
		try:
			categorized_series = ensure_categorical(
				df[var],
				method="quantile" if method == "Quantile" else ("width" if method == "Equal width" else "custom"),
				n_bins=int(bins),
				custom_edges=custom_edges_text,
			)
			df = df.copy()
			df[binned_col] = categorized_series
			st.success(f"Created categorical variable: {binned_col}")
		except Exception as exc:
			st.error(f"Categorization failed: {exc}")

	return df, (binned_col if binned_col in df.columns else var)


def table_generation_ui(df: pd.DataFrame, row_var: str, col_var: str) -> None:
	st.markdown("---")
	st.header("3) Contingency Table")
	if not row_var or not col_var:
		st.info("Select both row and column variables to generate a contingency table.")
		return

	with st.expander("Options"):
		normalize = st.checkbox("Normalize to proportions", value=False)
		dropna_levels = st.checkbox("Drop NA levels", value=True)
		min_count = st.number_input("Minimum count per cell to keep (0 = no filter)", min_value=0, max_value=100000, value=0)

	try:
		table = build_contingency_table(
			df,
			row_field=row_var,
			col_field=col_var,
			normalize=normalize,
			dropna_levels=dropna_levels,
			min_count=min_count,
		)
		st.dataframe(table, use_container_width=True)
		valid, notes = validate_contingency_table(table)
		if valid:
			st.success("Contingency table validated for statistical analysis.")
		else:
			st.warning("Validation warnings:")
			for n in notes:
				st.write(f"- {n}")

		csv = table.to_csv(index=True).encode("utf-8")
		st.download_button("Download CSV", data=csv, file_name="contingency_table.csv", mime="text/csv")
	except Exception as exc:
		st.error(f"Failed to build table: {exc}")


def main() -> None:
	init_state()
	st.title("CONFIRM Data Converter")
	st.caption("Transform raw client data into validated contingency tables for CONFIRM.")

	uploaded_df = sidebar_import()
	if uploaded_df is not None:
		st.session_state["source_df"] = uploaded_df
		st.session_state["working_df"] = uploaded_df.copy()

	df = st.session_state.get("working_df")
	if df is None or not isinstance(df, pd.DataFrame) or df.empty:
		st.info("Import data to begin.")
		return

	schema_and_preview(df)
	row_var, col_var = variable_selection_ui(df)

	if row_var:
		df, row_var = maybe_categorize(df, row_var, key_prefix="row")
	if col_var:
		df, col_var = maybe_categorize(df, col_var, key_prefix="col")

	if (row_var != st.session_state.get("row_var")) or (col_var != st.session_state.get("col_var")):
		st.session_state["row_var"] = row_var
		st.session_state["col_var"] = col_var
		st.session_state["working_df"] = df

	table_generation_ui(df, row_var, col_var)


if __name__ == "__main__":
	main()

