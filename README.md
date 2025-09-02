## CONFIRM Data Converter

Streamlit app to transform raw client data into validated contingency tables for CONFIRM.

### Features
- Import CSV, Excel, JSON, or query SQL databases
- Data preview (first 100 rows)
- Schema detection and field statistics
- Variable selection with recommendations
- Binning/categorization for continuous variables
- Contingency table generation and validation
- CSV export

### Setup
1. Python 3.9+
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Run
```bash
streamlit run app.py
```

### SQL Connection URL examples
- SQLite: `sqlite:///path/to/my.db`
- Postgres: `postgresql+psycopg2://user:pass@host:5432/dbname`
- MySQL: `mysql+pymysql://user:pass@host:3306/dbname`

