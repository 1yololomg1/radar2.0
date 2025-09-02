"""
Data Import Module
Handles importing data from various sources
"""

import pandas as pd
import json
from typing import Optional, Union
from io import StringIO, BytesIO
from sqlalchemy import create_engine, text
import streamlit as st


class DataImporter:
    """Class for importing data from various sources"""
    
    def __init__(self):
        """Initialize the data importer"""
        self.supported_formats = ['csv', 'excel', 'json', 'sql']
    
    def import_csv(self, file_or_path: Union[str, BytesIO, StringIO]) -> pd.DataFrame:
        """
        Import data from CSV file
        
        Args:
            file_or_path: File path or file object
            
        Returns:
            DataFrame containing the imported data
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_or_path, encoding=encoding)
                    
                    # Reset file pointer if it's a file object
                    if hasattr(file_or_path, 'seek'):
                        file_or_path.seek(0)
                    
                    return self._clean_dataframe(df)
                except UnicodeDecodeError:
                    if hasattr(file_or_path, 'seek'):
                        file_or_path.seek(0)
                    continue
            
            raise ValueError("Could not decode CSV file with any supported encoding")
            
        except Exception as e:
            raise ImportError(f"Error importing CSV: {str(e)}")
    
    def import_excel(self, file_or_path: Union[str, BytesIO]) -> pd.DataFrame:
        """
        Import data from Excel file
        
        Args:
            file_or_path: File path or file object
            
        Returns:
            DataFrame containing the imported data
        """
        try:
            # Check if multiple sheets exist
            excel_file = pd.ExcelFile(file_or_path)
            
            if len(excel_file.sheet_names) > 1:
                # Let user select sheet
                sheet_name = st.selectbox(
                    "Select sheet to import",
                    excel_file.sheet_names,
                    help="Multiple sheets detected. Please select one."
                )
                df = pd.read_excel(file_or_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_or_path)
            
            return self._clean_dataframe(df)
            
        except Exception as e:
            raise ImportError(f"Error importing Excel: {str(e)}")
    
    def import_json(self, file_or_path: Union[str, BytesIO, StringIO]) -> pd.DataFrame:
        """
        Import data from JSON file
        
        Args:
            file_or_path: File path or file object
            
        Returns:
            DataFrame containing the imported data
        """
        try:
            if isinstance(file_or_path, (BytesIO, StringIO)):
                content = file_or_path.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                data = json.loads(content)
            else:
                with open(file_or_path, 'r') as f:
                    data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Check if it's a records-oriented dict
                if all(isinstance(v, list) for v in data.values()):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame([data])
            else:
                raise ValueError("Unsupported JSON structure")
            
            return self._clean_dataframe(df)
            
        except Exception as e:
            raise ImportError(f"Error importing JSON: {str(e)}")
    
    def import_from_database(self, connection_string: str, query: str) -> pd.DataFrame:
        """
        Import data from database using SQL query
        
        Args:
            connection_string: Database connection string
            query: SQL query to execute
            
        Returns:
            DataFrame containing the query results
        """
        try:
            # Create engine
            engine = create_engine(connection_string)
            
            # Execute query and fetch data
            with engine.connect() as connection:
                df = pd.read_sql_query(text(query), connection)
            
            return self._clean_dataframe(df)
            
        except Exception as e:
            raise ImportError(f"Error importing from database: {str(e)}")
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare DataFrame after import
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Remove completely empty rows and columns
        df = df.dropna(how='all', axis=0)
        df = df.dropna(how='all', axis=1)
        
        # Strip whitespace from string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Convert obvious numeric columns
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Try to convert to numeric
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    # If more than 50% converted successfully, keep as numeric
                    if numeric_series.notna().sum() / len(df) > 0.5:
                        df[col] = numeric_series
                except:
                    pass
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        Validate imported data
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'info': []
        }
        
        # Check if DataFrame is empty
        if df.empty:
            validation['is_valid'] = False
            validation['errors'].append("DataFrame is empty")
            return validation
        
        # Check for minimum rows
        if len(df) < 10:
            validation['warnings'].append(f"Only {len(df)} rows found. Contingency tables may not be meaningful.")
        
        # Check for duplicate column names
        if df.columns.duplicated().any():
            validation['errors'].append("Duplicate column names detected")
            validation['is_valid'] = False
        
        # Check for columns with all missing values
        null_columns = df.columns[df.isnull().all()].tolist()
        if null_columns:
            validation['warnings'].append(f"Columns with all null values: {', '.join(null_columns)}")
        
        # Check for columns with single unique value
        single_value_cols = [col for col in df.columns if df[col].nunique() == 1]
        if single_value_cols:
            validation['warnings'].append(f"Columns with single value: {', '.join(single_value_cols)}")
        
        # Info about the data
        validation['info'].append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        validation['info'].append(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return validation
    
    def get_preview(self, df: pd.DataFrame, n_rows: int = 100) -> pd.DataFrame:
        """
        Get preview of the DataFrame
        
        Args:
            df: DataFrame to preview
            n_rows: Number of rows to show
            
        Returns:
            Preview DataFrame
        """
        return df.head(n_rows)