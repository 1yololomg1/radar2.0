"""
CONFIRM Data Converter
A tool to transform raw client data into properly formatted contingency tables
suitable for statistical validation with CONFIRM.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlalchemy as sa
from sqlalchemy import create_engine, text
import json
import io
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="CONFIRM Data Converter",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DataImportModule:
    """Handles data import from various sources and formats."""
    
    def __init__(self):
        self.data = None
        self.schema_info = {}
        self.connection = None
    
    def import_excel(self, file) -> pd.DataFrame:
        """Import data from Excel file."""
        try:
            # Try to read all sheets first
            excel_file = pd.ExcelFile(file)
            sheet_names = excel_file.sheet_names
            
            if len(sheet_names) > 1:
                sheet_name = st.selectbox(
                    "Select sheet to import:",
                    sheet_names,
                    key="excel_sheet_selector"
                )
            else:
                sheet_name = sheet_names[0]
            
            df = pd.read_excel(file, sheet_name=sheet_name)
            return df
        except Exception as e:
            st.error(f"Error importing Excel file: {str(e)}")
            return None
    
    def import_csv(self, file, encoding='utf-8') -> pd.DataFrame:
        """Import data from CSV file."""
        try:
            # Try different encodings if the first one fails
            encodings = [encoding, 'latin-1', 'cp1252', 'iso-8859-1']
            
            for enc in encodings:
                try:
                    file.seek(0)  # Reset file pointer
                    df = pd.read_csv(file, encoding=enc)
                    if enc != encoding:
                        st.info(f"File imported with {enc} encoding")
                    return df
                except UnicodeDecodeError:
                    continue
            
            st.error("Could not decode CSV file with any standard encoding")
            return None
        except Exception as e:
            st.error(f"Error importing CSV file: {str(e)}")
            return None
    
    def import_json(self, file) -> pd.DataFrame:
        """Import data from JSON file."""
        try:
            content = file.read()
            data = json.loads(content)
            
            # Handle different JSON structures
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Try to find the main data array
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        df = pd.DataFrame(value)
                        break
                else:
                    # If no list found, try to convert dict to DataFrame
                    df = pd.DataFrame([data])
            else:
                st.error("Unsupported JSON structure")
                return None
            
            return df
        except Exception as e:
            st.error(f"Error importing JSON file: {str(e)}")
            return None
    
    def connect_database(self, connection_string: str, query: str = None) -> pd.DataFrame:
        """Connect to database and execute query."""
        try:
            engine = create_engine(connection_string)
            
            if query:
                df = pd.read_sql(query, engine)
            else:
                # If no query provided, show available tables
                inspector = sa.inspect(engine)
                tables = inspector.get_table_names()
                
                if not tables:
                    st.error("No tables found in database")
                    return None
                
                table_name = st.selectbox("Select table:", tables)
                df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
            
            self.connection = engine
            return df
        except Exception as e:
            st.error(f"Database connection error: {str(e)}")
            return None
    
    def preview_data(self, df: pd.DataFrame, rows: int = 100) -> None:
        """Display data preview."""
        if df is None or df.empty:
            st.warning("No data to preview")
            return
        
        st.subheader("Data Preview")
        
        # Show basic info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        with col4:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Show first N rows
        st.write(f"**First {min(rows, len(df))} rows:**")
        st.dataframe(df.head(rows), use_container_width=True)
        
        # Show data types
        st.subheader("Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    def detect_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect and analyze data schema."""
        if df is None or df.empty:
            return {}
        
        schema_info = {}
        
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'null_count': df[col].isnull().sum(),
                'unique_count': df[col].nunique(),
                'is_categorical': False,
                'is_continuous': False,
                'suggested_type': 'unknown'
            }
            
            # Determine if column is categorical or continuous
            if df[col].dtype in ['object', 'category', 'string']:
                col_info['is_categorical'] = True
                col_info['suggested_type'] = 'categorical'
            elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                unique_ratio = col_info['unique_count'] / len(df)
                if unique_ratio < 0.1:  # Less than 10% unique values
                    col_info['is_categorical'] = True
                    col_info['suggested_type'] = 'categorical'
                else:
                    col_info['is_continuous'] = True
                    col_info['suggested_type'] = 'continuous'
            
            # Add value distribution for categorical columns
            if col_info['is_categorical']:
                col_info['value_counts'] = df[col].value_counts().head(10).to_dict()
            
            # Add statistics for continuous columns
            if col_info['is_continuous']:
                col_info['min'] = df[col].min()
                col_info['max'] = df[col].max()
                col_info['mean'] = df[col].mean()
                col_info['std'] = df[col].std()
            
            schema_info[col] = col_info
        
        self.schema_info = schema_info
        return schema_info

class VariableSelectionInterface:
    """Handles variable selection and analysis for contingency tables."""
    
    def __init__(self, schema_info: Dict[str, Any]):
        self.schema_info = schema_info
        self.row_var = None
        self.col_var = None
    
    def display_variable_analysis(self, df: pd.DataFrame) -> None:
        """Display detailed analysis of each variable."""
        st.subheader("Variable Analysis")
        
        for col, info in self.schema_info.items():
            with st.expander(f"**{col}** ({info['suggested_type']})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Data Type:** {info['dtype']}")
                    st.write(f"**Null Values:** {info['null_count']}")
                    st.write(f"**Unique Values:** {info['unique_count']}")
                
                with col2:
                    if info['is_categorical'] and 'value_counts' in info:
                        st.write("**Top Values:**")
                        for val, count in list(info['value_counts'].items())[:5]:
                            st.write(f"  • {val}: {count}")
                    elif info['is_continuous']:
                        st.write("**Statistics:**")
                        st.write(f"  • Min: {info['min']:.2f}")
                        st.write(f"  • Max: {info['max']:.2f}")
                        st.write(f"  • Mean: {info['mean']:.2f}")
                        st.write(f"  • Std: {info['std']:.2f}")
    
    def select_variables(self, df: pd.DataFrame) -> Tuple[str, str]:
        """Interface for selecting row and column variables."""
        st.subheader("Variable Selection")
        
        # Get categorical columns
        categorical_cols = [col for col, info in self.schema_info.items() 
                          if info['suggested_type'] == 'categorical']
        continuous_cols = [col for col, info in self.schema_info.items() 
                          if info['suggested_type'] == 'continuous']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Row Variable (X-axis):**")
            self.row_var = st.selectbox(
                "Select row variable:",
                options=[''] + list(df.columns),
                key="row_var_selector"
            )
            
            if self.row_var and self.row_var in continuous_cols:
                st.warning("Continuous variable selected. Consider categorization.")
        
        with col2:
            st.write("**Column Variable (Y-axis):**")
            self.col_var = st.selectbox(
                "Select column variable:",
                options=[''] + list(df.columns),
                key="col_var_selector"
            )
            
            if self.col_var and self.col_var in continuous_cols:
                st.warning("Continuous variable selected. Consider categorization.")
        
        # Recommendations
        if categorical_cols:
            st.info(f"**Recommended categorical variables:** {', '.join(categorical_cols[:5])}")
        
        return self.row_var, self.col_var
    
    def get_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate variable recommendations based on cardinality and distribution."""
        recommendations = []
        
        for col, info in self.schema_info.items():
            if info['suggested_type'] == 'categorical':
                # Good for contingency tables: 2-20 unique values
                if 2 <= info['unique_count'] <= 20:
                    recommendations.append(col)
        
        return recommendations

class ContingencyTableGenerator:
    """Generates and validates contingency tables."""
    
    def __init__(self):
        self.contingency_table = None
        self.validation_results = {}
    
    def create_contingency_table(self, df: pd.DataFrame, row_var: str, col_var: str) -> pd.DataFrame:
        """Create contingency table from selected variables."""
        if not row_var or not col_var:
            st.error("Please select both row and column variables")
            return None
        
        try:
            # Create contingency table
            contingency_table = pd.crosstab(df[row_var], df[col_var], margins=True)
            self.contingency_table = contingency_table
            
            return contingency_table
        except Exception as e:
            st.error(f"Error creating contingency table: {str(e)}")
            return None
    
    def validate_table(self, contingency_table: pd.DataFrame) -> Dict[str, Any]:
        """Validate contingency table for CONFIRM requirements."""
        if contingency_table is None:
            return {}
        
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check minimum cell counts
        min_cell_count = 5
        low_cells = (contingency_table < min_cell_count).sum().sum()
        
        if low_cells > 0:
            validation['warnings'].append(f"{low_cells} cells have counts < {min_cell_count}")
            validation['recommendations'].append("Consider combining categories with low counts")
        
        # Check table dimensions
        rows, cols = contingency_table.shape
        if rows < 2 or cols < 2:
            validation['errors'].append("Table must have at least 2 rows and 2 columns")
            validation['is_valid'] = False
        
        # Check for empty rows/columns
        empty_rows = (contingency_table.sum(axis=1) == 0).sum()
        empty_cols = (contingency_table.sum(axis=0) == 0).sum()
        
        if empty_rows > 0:
            validation['warnings'].append(f"{empty_rows} empty rows found")
        if empty_cols > 0:
            validation['warnings'].append(f"{empty_cols} empty columns found")
        
        self.validation_results = validation
        return validation
    
    def display_table(self, contingency_table: pd.DataFrame) -> None:
        """Display contingency table with formatting."""
        if contingency_table is None:
            return
        
        st.subheader("Contingency Table")
        
        # Display the table
        st.dataframe(contingency_table, use_container_width=True)
        
        # Display validation results
        if self.validation_results:
            st.subheader("Validation Results")
            
            if self.validation_results['is_valid']:
                st.success("Table is valid for CONFIRM analysis")
            else:
                st.error("Table has validation errors")
            
            if self.validation_results['warnings']:
                st.warning("Warnings:")
                for warning in self.validation_results['warnings']:
                    st.write(f"  • {warning}")
            
            if self.validation_results['errors']:
                st.error("Errors:")
                for error in self.validation_results['errors']:
                    st.write(f"  • {error}")
            
            if self.validation_results['recommendations']:
                st.info("Recommendations:")
                for rec in self.validation_results['recommendations']:
                    st.write(f"  • {rec}")

def main():
    """Main application function."""
    st.title("CONFIRM Data Converter")
    st.markdown("Transform raw client data into properly formatted contingency tables for statistical validation")
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'schema_info' not in st.session_state:
        st.session_state.schema_info = {}
    
    # Sidebar for data import
    with st.sidebar:
        st.header("Data Import")
        
        import_type = st.radio(
            "Select import method:",
            ["Upload File", "Database Connection"]
        )
        
        if import_type == "Upload File":
            file_type = st.selectbox(
                "File type:",
                ["Excel (.xlsx)", "CSV (.csv)", "JSON (.json)"]
            )
            
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['xlsx', 'csv', 'json']
            )
            
            if uploaded_file is not None:
                data_importer = DataImportModule()
                
                if file_type.startswith("Excel"):
                    df = data_importer.import_excel(uploaded_file)
                elif file_type.startswith("CSV"):
                    df = data_importer.import_csv(uploaded_file)
                elif file_type.startswith("JSON"):
                    df = data_importer.import_json(uploaded_file)
                
                if df is not None:
                    st.session_state.data = df
                    st.session_state.schema_info = data_importer.detect_schema(df)
                    st.success("Data imported successfully!")
        
        else:  # Database Connection
            st.subheader("Database Connection")
            
            db_type = st.selectbox(
                "Database type:",
                ["PostgreSQL", "MySQL", "SQLite", "SQL Server"]
            )
            
            if db_type == "PostgreSQL":
                connection_string = st.text_input(
                    "Connection string:",
                    placeholder="postgresql://user:password@host:port/database"
                )
            elif db_type == "MySQL":
                connection_string = st.text_input(
                    "Connection string:",
                    placeholder="mysql+pymysql://user:password@host:port/database"
                )
            elif db_type == "SQLite":
                connection_string = st.text_input(
                    "Connection string:",
                    placeholder="sqlite:///path/to/database.db"
                )
            else:  # SQL Server
                connection_string = st.text_input(
                    "Connection string:",
                    placeholder="mssql+pyodbc://user:password@host:port/database?driver=ODBC+Driver+17+for+SQL+Server"
                )
            
            custom_query = st.text_area(
                "Custom SQL query (optional):",
                placeholder="SELECT * FROM table_name WHERE condition"
            )
            
            if st.button("Connect to Database"):
                if connection_string:
                    data_importer = DataImportModule()
                    df = data_importer.connect_database(connection_string, custom_query)
                    
                    if df is not None:
                        st.session_state.data = df
                        st.session_state.schema_info = data_importer.detect_schema(df)
                        st.success("Database connected successfully!")
    
    # Main content area
    if st.session_state.data is not None:
        df = st.session_state.data
        schema_info = st.session_state.schema_info
        
        # Data preview
        data_importer = DataImportModule()
        data_importer.preview_data(df)
        
        # Variable analysis
        var_interface = VariableSelectionInterface(schema_info)
        var_interface.display_variable_analysis(df)
        
        # Variable selection
        row_var, col_var = var_interface.select_variables(df)
        
        # Generate contingency table
        if row_var and col_var:
            table_generator = ContingencyTableGenerator()
            contingency_table = table_generator.create_contingency_table(df, row_var, col_var)
            
            if contingency_table is not None:
                # Validate table
                validation_results = table_generator.validate_table(contingency_table)
                
                # Display results
                table_generator.display_table(contingency_table)
                
                # Download option
                st.subheader("Download Results")
                
                # Convert to CSV for download
                csv = contingency_table.to_csv()
                st.download_button(
                    label="Download Contingency Table (CSV)",
                    data=csv,
                    file_name=f"contingency_table_{row_var}_vs_{col_var}.csv",
                    mime="text/csv"
                )
                
                # Convert to Excel for download
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    contingency_table.to_excel(writer, sheet_name='Contingency_Table')
                    
                    # Add validation results sheet
                    validation_df = pd.DataFrame({
                        'Validation Item': ['Is Valid', 'Warnings', 'Errors', 'Recommendations'],
                        'Result': [
                            validation_results.get('is_valid', False),
                            '; '.join(validation_results.get('warnings', [])),
                            '; '.join(validation_results.get('errors', [])),
                            '; '.join(validation_results.get('recommendations', []))
                        ]
                    })
                    validation_df.to_excel(writer, sheet_name='Validation_Results', index=False)
                
                st.download_button(
                    label="Download Full Report (Excel)",
                    data=output.getvalue(),
                    file_name=f"confirm_report_{row_var}_vs_{col_var}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    else:
        st.info("Please import data using the sidebar to get started")
        
        # Show example
        st.subheader("Example Usage")
        st.markdown("""
        1. **Import Data**: Use the sidebar to upload a file or connect to a database
        2. **Review Schema**: The system will automatically detect data types and suggest categorical variables
        3. **Select Variables**: Choose row and column variables for your contingency table
        4. **Generate Table**: Create and validate the contingency table
        5. **Download Results**: Export the table and validation report
        
        **Supported Formats:**
        - Excel files (.xlsx)
        - CSV files (.csv)
        - JSON files (.json)
        - Database connections (PostgreSQL, MySQL, SQLite, SQL Server)
        """)

if __name__ == "__main__":
    main()