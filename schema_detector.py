"""
Schema Detection Module
Automatically identifies data types and potential categorical fields
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime


class SchemaDetector:
    """Class for detecting and analyzing data schema"""
    
    def __init__(self):
        """Initialize the schema detector"""
        self.categorical_threshold = 0.5  # Max unique ratio for categorical
        self.text_threshold = 0.95  # Min unique ratio for text
        self.max_categories = 50  # Maximum unique values for categorical
    
    def detect_schema(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Detect schema of the DataFrame
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with schema information for each column
        """
        schema = {}
        
        for column in df.columns:
            schema[column] = self._analyze_column(df[column], column)
        
        return schema
    
    def _analyze_column(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """
        Analyze a single column
        
        Args:
            series: Column data
            column_name: Name of the column
            
        Returns:
            Dictionary with column analysis
        """
        analysis = {
            'data_type': None,
            'nullable': series.isnull().any(),
            'unique_values': series.nunique(),
            'sample_values': []
        }
        
        # Get non-null values for analysis
        non_null = series.dropna()
        
        if len(non_null) == 0:
            analysis['data_type'] = 'empty'
            return analysis
        
        # Get sample values
        unique_vals = non_null.unique()
        analysis['sample_values'] = list(unique_vals[:5])
        
        # Determine data type
        if self._is_datetime(non_null):
            analysis['data_type'] = 'datetime'
            analysis['min_date'] = non_null.min()
            analysis['max_date'] = non_null.max()
        
        elif self._is_numeric(non_null):
            if self._should_be_categorical(non_null, is_numeric=True):
                analysis['data_type'] = 'categorical'
                analysis['categories'] = sorted(unique_vals.tolist())
            else:
                analysis['data_type'] = 'numeric'
                analysis['min'] = float(non_null.min())
                analysis['max'] = float(non_null.max())
                analysis['mean'] = float(non_null.mean())
                analysis['std'] = float(non_null.std())
        
        elif self._is_boolean(non_null):
            analysis['data_type'] = 'boolean'
            analysis['categories'] = sorted(unique_vals.tolist())
        
        else:  # String/Object type
            if self._should_be_categorical(non_null, is_numeric=False):
                analysis['data_type'] = 'categorical'
                if len(unique_vals) <= self.max_categories:
                    analysis['categories'] = sorted(unique_vals.tolist())
                else:
                    analysis['categories'] = sorted(unique_vals[:self.max_categories].tolist())
                    analysis['truncated'] = True
            else:
                analysis['data_type'] = 'text'
                analysis['avg_length'] = non_null.astype(str).str.len().mean()
        
        return analysis
    
    def _is_datetime(self, series: pd.Series) -> bool:
        """
        Check if series contains datetime values
        
        Args:
            series: Series to check
            
        Returns:
            True if datetime, False otherwise
        """
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        # Try to parse as datetime
        if series.dtype == 'object':
            try:
                pd.to_datetime(series.iloc[:100], errors='coerce')
                parsed = pd.to_datetime(series, errors='coerce')
                # If more than 50% parsed successfully, consider it datetime
                return parsed.notna().sum() / len(series) > 0.5
            except:
                return False
        
        return False
    
    def _is_numeric(self, series: pd.Series) -> bool:
        """
        Check if series contains numeric values
        
        Args:
            series: Series to check
            
        Returns:
            True if numeric, False otherwise
        """
        return pd.api.types.is_numeric_dtype(series)
    
    def _is_boolean(self, series: pd.Series) -> bool:
        """
        Check if series contains boolean values
        
        Args:
            series: Series to check
            
        Returns:
            True if boolean, False otherwise
        """
        if pd.api.types.is_bool_dtype(series):
            return True
        
        # Check if it's effectively boolean (e.g., Yes/No, True/False as strings)
        unique_vals = series.unique()
        if len(unique_vals) <= 3:  # Allow for some nulls
            str_vals = set(str(v).lower() for v in unique_vals if pd.notna(v))
            boolean_sets = [
                {'yes', 'no'},
                {'true', 'false'},
                {'1', '0'},
                {'y', 'n'},
                {'t', 'f'}
            ]
            return any(str_vals.issubset(bool_set) for bool_set in boolean_sets)
        
        return False
    
    def _should_be_categorical(self, series: pd.Series, is_numeric: bool) -> bool:
        """
        Determine if a series should be treated as categorical
        
        Args:
            series: Series to check
            is_numeric: Whether the series is numeric
            
        Returns:
            True if should be categorical, False otherwise
        """
        unique_ratio = series.nunique() / len(series)
        
        # For numeric data
        if is_numeric:
            # Check if it's likely an ID column
            if unique_ratio > 0.95:
                return False
            
            # Check if it looks like discrete categories
            unique_count = series.nunique()
            if unique_count <= 20:  # Small number of unique values
                return True
            
            # Check if values are integers (possible categories)
            if series.dtype in ['int64', 'int32', 'int16', 'int8']:
                if unique_count <= self.max_categories:
                    # Check if values are sequential or have gaps
                    sorted_unique = np.sort(series.unique())
                    if len(sorted_unique) > 1:
                        gaps = np.diff(sorted_unique)
                        # If gaps are not uniform, likely categorical
                        if np.std(gaps) > np.mean(gaps) * 0.5:
                            return True
            
            return False
        
        # For non-numeric data
        else:
            # High cardinality suggests text/ID field
            if unique_ratio > self.text_threshold:
                return False
            
            # Low cardinality suggests categorical
            if series.nunique() <= self.max_categories:
                return True
            
            # Medium cardinality - check string patterns
            sample = series.head(100)
            avg_length = sample.astype(str).str.len().mean()
            
            # Short strings are more likely categorical
            if avg_length < 30:
                return True
            
            return False
    
    def get_categorical_columns(self, schema: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Get list of categorical columns from schema
        
        Args:
            schema: Schema dictionary
            
        Returns:
            List of categorical column names
        """
        return [col for col, info in schema.items() 
                if info['data_type'] in ['categorical', 'boolean']]
    
    def get_numeric_columns(self, schema: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Get list of numeric columns from schema
        
        Args:
            schema: Schema dictionary
            
        Returns:
            List of numeric column names
        """
        return [col for col, info in schema.items() 
                if info['data_type'] == 'numeric']
    
    def suggest_categorization(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Suggest categorization strategy for a continuous variable
        
        Args:
            df: DataFrame
            column: Column name
            
        Returns:
            Dictionary with categorization suggestions
        """
        series = df[column].dropna()
        
        suggestions = {
            'column': column,
            'current_type': 'numeric',
            'strategies': []
        }
        
        # Equal width binning
        suggestions['strategies'].append({
            'method': 'equal_width',
            'n_bins': self._suggest_n_bins(series),
            'description': 'Divide range into equal-width intervals'
        })
        
        # Equal frequency binning
        suggestions['strategies'].append({
            'method': 'equal_frequency',
            'n_bins': self._suggest_n_bins(series),
            'description': 'Divide data into bins with equal number of observations'
        })
        
        # Natural breaks (if applicable)
        if len(series) > 100:
            suggestions['strategies'].append({
                'method': 'natural_breaks',
                'n_bins': self._suggest_n_bins(series),
                'description': 'Use natural groupings in the data'
            })
        
        # Domain-specific suggestions
        if 'age' in column.lower():
            suggestions['strategies'].append({
                'method': 'custom',
                'bins': [0, 18, 30, 45, 60, 100],
                'labels': ['<18', '18-29', '30-44', '45-59', '60+'],
                'description': 'Age groups commonly used in demographics'
            })
        
        elif 'income' in column.lower() or 'salary' in column.lower():
            suggestions['strategies'].append({
                'method': 'custom',
                'bins': [0, 30000, 60000, 100000, float('inf')],
                'labels': ['<30k', '30-60k', '60-100k', '>100k'],
                'description': 'Income brackets'
            })
        
        return suggestions
    
    def _suggest_n_bins(self, series: pd.Series) -> int:
        """
        Suggest number of bins for categorization
        
        Args:
            series: Series to categorize
            
        Returns:
            Suggested number of bins
        """
        n = len(series)
        
        # Sturges' rule
        sturges = int(np.ceil(np.log2(n) + 1))
        
        # Square root rule
        sqrt_rule = int(np.ceil(np.sqrt(n)))
        
        # Take average and cap at reasonable limits
        suggested = min(max(3, (sturges + sqrt_rule) // 2), 10)
        
        return suggested