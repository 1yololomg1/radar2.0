#!/usr/bin/env python3
"""
CONFIRM Data Converter
Transform raw client data into properly formatted contingency tables for statistical validation

This module provides comprehensive data import, variable selection, and contingency table
generation capabilities for the CONFIRM statistical validation system.

Version: 1.0.0
Author: CONFIRM Development Team
"""

import sys
import os
import logging
import traceback
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import json
import sqlite3

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns

from scipy.stats import chi2_contingency, chi2, normaltest, skew, kurtosis
import openpyxl
from openpyxl import load_workbook

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DataProfile:
    """Profile of a dataset column for analysis and recommendations"""
    name: str
    dtype: str
    unique_count: int
    null_count: int
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    sample_values: List[Any] = field(default_factory=list)
    is_categorical: bool = False
    is_continuous: bool = False
    cardinality_ratio: float = 0.0
    recommended_for_contingency: bool = False


@dataclass
class ContingencyTableConfig:
    """Configuration for generating contingency tables"""
    row_variable: str
    column_variable: str
    row_categories: Optional[List[str]] = None
    column_categories: Optional[List[str]] = None
    bin_continuous: bool = False
    num_bins: int = 5
    custom_bins: Optional[List[float]] = None
    categorization_method: str = 'equal_width'


class DataImportModule:
    """Handles importing data from various file formats and databases"""
    
    def __init__(self):
        self.supported_formats = {
            '.csv': self._read_csv,
            '.xlsx': self._read_excel,
            '.xls': self._read_excel,
            '.json': self._read_json,
            '.db': self._read_sqlite,
            '.sqlite': self._read_sqlite,
            '.sqlite3': self._read_sqlite
        }
        
    def import_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Import data from supported file formats"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            logger.info(f"Importing data from {file_path}")
            df = self.supported_formats[file_ext](file_path, **kwargs)
            
            logger.info(f"Successfully imported {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error importing data: {str(e)}")
            raise
    
    def _read_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read CSV file with automatic encoding detection"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                return pd.read_csv(file_path, encoding=encoding, **kwargs)
            except UnicodeDecodeError:
                continue
        
        raise ValueError("Could not read CSV file with any supported encoding")
    
    def _read_excel(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read Excel file"""
        return pd.read_excel(file_path, **kwargs)
    
    def _read_json(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        else:
            raise ValueError("JSON file must contain a list or dictionary")
    
    def _read_sqlite(self, file_path: str, query: str = None, **kwargs) -> pd.DataFrame:
        """Read from SQLite database"""
        conn = sqlite3.connect(file_path)
        
        try:
            if query is None:
                # Get list of tables
                tables = pd.read_sql_query(
                    "SELECT name FROM sqlite_master WHERE type='table'", 
                    conn
                )
                if len(tables) == 0:
                    raise ValueError("No tables found in database")
                
                # Use first table if no query specified
                table_name = tables.iloc[0]['name']
                query = f"SELECT * FROM {table_name}"
            
            return pd.read_sql_query(query, conn, **kwargs)
        
        finally:
            conn.close()
    
    def get_preview(self, df: pd.DataFrame, num_rows: int = 100) -> pd.DataFrame:
        """Get preview of the first N rows"""
        return df.head(num_rows)
    
    def detect_schema(self, df: pd.DataFrame) -> Dict[str, DataProfile]:
        """Automatically detect schema and create data profiles"""
        profiles = {}
        total_rows = len(df)
        
        for col in df.columns:
            series = df[col]
            
            # Basic statistics
            unique_count = series.nunique()
            null_count = series.isnull().sum()
            cardinality_ratio = unique_count / total_rows if total_rows > 0 else 0
            
            # Sample values (non-null)
            sample_values = series.dropna().head(10).tolist()
            
            # Determine if categorical or continuous
            is_categorical = False
            is_continuous = False
            min_val = None
            max_val = None
            
            if pd.api.types.is_numeric_dtype(series):
                min_val = series.min() if not series.empty else None
                max_val = series.max() if not series.empty else None
                
                # Consider categorical if low cardinality or all integers in small range
                if cardinality_ratio < 0.05 or (unique_count <= 20 and series.dtype in ['int64', 'int32']):
                    is_categorical = True
                else:
                    is_continuous = True
            else:
                is_categorical = True
            
            # Recommendation for contingency tables
            recommended = (
                is_categorical and 
                unique_count >= 2 and 
                unique_count <= 50 and 
                null_count / total_rows < 0.5
            )
            
            profiles[col] = DataProfile(
                name=col,
                dtype=str(series.dtype),
                unique_count=unique_count,
                null_count=null_count,
                min_value=min_val,
                max_value=max_val,
                sample_values=sample_values,
                is_categorical=is_categorical,
                is_continuous=is_continuous,
                cardinality_ratio=cardinality_ratio,
                recommended_for_contingency=recommended
            )
        
        return profiles


class VariableSelectionInterface:
    """Interface for selecting and configuring variables for contingency table analysis"""
    
    def __init__(self, parent_frame: tk.Frame):
        self.parent_frame = parent_frame
        self.data_profiles: Dict[str, DataProfile] = {}
        self.selected_row_var = tk.StringVar()
        self.selected_col_var = tk.StringVar()
        
        self._create_interface()
    
    def _create_interface(self):
        """Create the variable selection interface"""
        # Main frame
        main_frame = ttk.LabelFrame(self.parent_frame, text="Variable Selection", padding="10")
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Variable selection frame
        selection_frame = ttk.Frame(main_frame)
        selection_frame.pack(fill='x', pady=(0, 10))
        
        # Row variable selection
        ttk.Label(selection_frame, text="Row Variable (X-axis):").grid(row=0, column=0, sticky='w', padx=(0, 10))
        self.row_var_combo = ttk.Combobox(selection_frame, textvariable=self.selected_row_var, 
                                         state='readonly', width=30)
        self.row_var_combo.grid(row=0, column=1, padx=(0, 20))
        self.row_var_combo.bind('<<ComboboxSelected>>', self._on_variable_selected)
        
        # Column variable selection
        ttk.Label(selection_frame, text="Column Variable (Y-axis):").grid(row=0, column=2, sticky='w', padx=(0, 10))
        self.col_var_combo = ttk.Combobox(selection_frame, textvariable=self.selected_col_var, 
                                         state='readonly', width=30)
        self.col_var_combo.grid(row=0, column=3)
        self.col_var_combo.bind('<<ComboboxSelected>>', self._on_variable_selected)
        
        # Statistics display frame
        stats_frame = ttk.LabelFrame(main_frame, text="Variable Statistics", padding="10")
        stats_frame.pack(fill='both', expand=True)
        
        # Create treeview for statistics
        columns = ('Variable', 'Type', 'Unique Values', 'Null Count', 'Range', 'Recommended')
        self.stats_tree = ttk.Treeview(stats_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.stats_tree.heading(col, text=col)
            self.stats_tree.column(col, width=120)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(stats_frame, orient='vertical', command=self.stats_tree.yview)
        self.stats_tree.configure(yscrollcommand=scrollbar.set)
        
        self.stats_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Recommendation frame
        rec_frame = ttk.LabelFrame(main_frame, text="Recommendations", padding="10")
        rec_frame.pack(fill='x', pady=(10, 0))
        
        self.recommendation_text = scrolledtext.ScrolledText(rec_frame, height=4, wrap=tk.WORD)
        self.recommendation_text.pack(fill='x')
    
    def update_data_profiles(self, profiles: Dict[str, DataProfile]):
        """Update the interface with new data profiles"""
        self.data_profiles = profiles
        
        # Update combobox options
        variable_names = list(profiles.keys())
        self.row_var_combo['values'] = variable_names
        self.col_var_combo['values'] = variable_names
        
        # Update statistics tree
        self._update_statistics_tree()
        
        # Generate recommendations
        self._generate_recommendations()
    
    def _update_statistics_tree(self):
        """Update the statistics treeview with current data profiles"""
        # Clear existing items
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
        
        # Add new items
        for profile in self.data_profiles.values():
            # Format range
            if profile.min_value is not None and profile.max_value is not None:
                range_str = f"{profile.min_value:.2f} - {profile.max_value:.2f}"
            else:
                range_str = "N/A"
            
            # Format type
            type_str = "Categorical" if profile.is_categorical else "Continuous"
            if profile.is_continuous and profile.unique_count <= 20:
                type_str += " (Low cardinality)"
            
            # Recommended status
            rec_str = "Yes" if profile.recommended_for_contingency else "No"
            
            self.stats_tree.insert('', 'end', values=(
                profile.name,
                type_str,
                profile.unique_count,
                profile.null_count,
                range_str,
                rec_str
            ))
    
    def _generate_recommendations(self):
        """Generate recommendations for variable selection"""
        recommendations = []
        
        # Find recommended variables
        recommended_vars = [p for p in self.data_profiles.values() if p.recommended_for_contingency]
        continuous_vars = [p for p in self.data_profiles.values() if p.is_continuous]
        
        if len(recommended_vars) >= 2:
            recommendations.append(f"✓ Found {len(recommended_vars)} variables suitable for contingency analysis:")
            for var in recommended_vars[:5]:  # Show top 5
                recommendations.append(f"  • {var.name} ({var.unique_count} categories)")
        
        if continuous_vars:
            recommendations.append(f"\n⚠ {len(continuous_vars)} continuous variables detected:")
            for var in continuous_vars[:3]:  # Show top 3
                recommendations.append(f"  • {var.name} (range: {var.min_value:.2f} - {var.max_value:.2f})")
            recommendations.append("  These will need categorization for contingency analysis.")
        
        if len(recommended_vars) < 2:
            recommendations.append("⚠ Warning: Less than 2 suitable categorical variables found.")
            recommendations.append("Consider categorizing continuous variables or checking data quality.")
        
        # Update recommendation text
        self.recommendation_text.delete(1.0, tk.END)
        self.recommendation_text.insert(1.0, '\n'.join(recommendations))
    
    def _on_variable_selected(self, event=None):
        """Handle variable selection events"""
        row_var = self.selected_row_var.get()
        col_var = self.selected_col_var.get()
        
        if row_var and col_var and row_var != col_var:
            # Variables selected, enable next steps
            self._update_selection_recommendations(row_var, col_var)
    
    def _update_selection_recommendations(self, row_var: str, col_var: str):
        """Update recommendations based on selected variables"""
        if row_var not in self.data_profiles or col_var not in self.data_profiles:
            return
        
        row_profile = self.data_profiles[row_var]
        col_profile = self.data_profiles[col_var]
        
        recommendations = [f"Selected Variables:"]
        recommendations.append(f"Row Variable: {row_var} ({row_profile.unique_count} categories)")
        recommendations.append(f"Column Variable: {col_var} ({col_profile.unique_count} categories)")
        
        # Check for issues
        issues = []
        if row_profile.is_continuous:
            issues.append(f"• {row_var} is continuous - categorization required")
        if col_profile.is_continuous:
            issues.append(f"• {col_var} is continuous - categorization required")
        
        if row_profile.unique_count > 20:
            issues.append(f"• {row_var} has many categories ({row_profile.unique_count}) - consider grouping")
        if col_profile.unique_count > 20:
            issues.append(f"• {col_var} has many categories ({col_profile.unique_count}) - consider grouping")
        
        if issues:
            recommendations.append("\nIssues to Address:")
            recommendations.extend(issues)
        else:
            recommendations.append("\n✓ Variables are suitable for contingency table analysis")
        
        # Update recommendation text
        self.recommendation_text.delete(1.0, tk.END)
        self.recommendation_text.insert(1.0, '\n'.join(recommendations))
    
    def get_selection_config(self) -> Optional[ContingencyTableConfig]:
        """Get the current variable selection configuration"""
        row_var = self.selected_row_var.get()
        col_var = self.selected_col_var.get()
        
        if not row_var or not col_var or row_var == col_var:
            return None
        
        return ContingencyTableConfig(
            row_variable=row_var,
            column_variable=col_var
        )


class CategorizationEngine:
    """Engine for categorizing continuous variables"""
    
    def __init__(self):
        self.categorization_methods = {
            'equal_width': self._equal_width_binning,
            'equal_frequency': self._equal_frequency_binning,
            'quantile': self._quantile_binning,
            'custom': self._custom_binning,
            'jenks': self._jenks_binning,
            'kmeans': self._kmeans_binning
        }
    
    def categorize_variable(self, series: pd.Series, method: str = 'equal_width', 
                          num_bins: int = 5, custom_bins: List[float] = None) -> pd.Series:
        """Categorize a continuous variable using specified method"""
        
        if method not in self.categorization_methods:
            raise ValueError(f"Unknown categorization method: {method}")
        
        # Remove null values for binning
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return pd.Series(index=series.index, dtype='category')
        
        # Apply categorization method
        categories = self.categorization_methods[method](non_null_series, num_bins, custom_bins)
        
        # Create categorical series
        result = pd.cut(series, bins=categories, include_lowest=True, duplicates='drop')
        
        return result
    
    def _equal_width_binning(self, series: pd.Series, num_bins: int, custom_bins: List[float] = None) -> np.ndarray:
        """Create equal-width bins"""
        min_val, max_val = series.min(), series.max()
        return np.linspace(min_val, max_val, num_bins + 1)
    
    def _equal_frequency_binning(self, series: pd.Series, num_bins: int, custom_bins: List[float] = None) -> np.ndarray:
        """Create equal-frequency bins"""
        quantiles = np.linspace(0, 1, num_bins + 1)
        return series.quantile(quantiles).values
    
    def _quantile_binning(self, series: pd.Series, num_bins: int, custom_bins: List[float] = None) -> np.ndarray:
        """Create quantile-based bins"""
        if num_bins == 4:
            quantiles = [0, 0.25, 0.5, 0.75, 1.0]
        elif num_bins == 5:
            quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        else:
            quantiles = np.linspace(0, 1, num_bins + 1)
        
        return series.quantile(quantiles).values
    
    def _custom_binning(self, series: pd.Series, num_bins: int, custom_bins: List[float] = None) -> np.ndarray:
        """Use custom bin edges"""
        if custom_bins is None:
            raise ValueError("Custom bins must be provided for custom binning method")
        
        return np.array(custom_bins)
    
    def _jenks_binning(self, series: pd.Series, num_bins: int, custom_bins: List[float] = None) -> np.ndarray:
        """Jenks natural breaks optimization (simplified version)"""
        # Simplified implementation - for full Jenks, would need additional library
        sorted_values = np.sort(series.values)
        n = len(sorted_values)
        
        if n <= num_bins:
            return np.concatenate([[sorted_values[0]], sorted_values, [sorted_values[-1]]])
        
        # Use quantile-based approximation
        indices = np.linspace(0, n-1, num_bins + 1, dtype=int)
        return sorted_values[indices]
    
    def _kmeans_binning(self, series: pd.Series, num_bins: int, custom_bins: List[float] = None) -> np.ndarray:
        """K-means based binning for natural groupings"""
        try:
            from sklearn.cluster import KMeans
            
            # Reshape for sklearn
            data = series.dropna().values.reshape(-1, 1)
            
            if len(data) < num_bins:
                return self._equal_width_binning(series, num_bins, custom_bins)
            
            # Fit K-means
            kmeans = KMeans(n_clusters=num_bins, random_state=42, n_init=10)
            kmeans.fit(data)
            
            # Get cluster centers and create bins
            centers = kmeans.cluster_centers_.flatten()
            centers = np.sort(centers)
            
            # Create bin edges
            min_val, max_val = series.min(), series.max()
            bin_edges = [min_val]
            
            for i in range(len(centers) - 1):
                bin_edges.append((centers[i] + centers[i + 1]) / 2)
            
            bin_edges.append(max_val)
            
            return np.array(bin_edges)
            
        except ImportError:
            logger.warning("scikit-learn not available, falling back to equal-width binning")
            return self._equal_width_binning(series, num_bins, custom_bins)
    
    def suggest_bins(self, series: pd.Series) -> Dict[str, Any]:
        """Suggest optimal number of bins and method for a continuous variable"""
        n = len(series.dropna())
        
        if n == 0:
            return {'method': 'equal_width', 'num_bins': 3, 'reason': 'No data available'}
        
        # Sturges' rule
        sturges_bins = int(np.ceil(np.log2(n) + 1))
        
        # Square root rule
        sqrt_bins = int(np.ceil(np.sqrt(n)))
        
        # Rice rule
        rice_bins = int(np.ceil(2 * (n ** (1/3))))
        
        # Choose optimal
        suggested_bins = min(max(sturges_bins, 3), 10)  # Between 3 and 10
        
        # Determine method based on data distribution
        skewness = series.skew()
        if abs(skewness) > 1:
            method = 'quantile'
            reason = f"Quantile binning recommended due to skewed distribution (skew={skewness:.2f})"
        else:
            method = 'equal_width'
            reason = "Equal-width binning suitable for relatively normal distribution"
        
        return {
            'method': method,
            'num_bins': suggested_bins,
            'sturges': sturges_bins,
            'sqrt': sqrt_bins,
            'rice': rice_bins,
            'reason': reason
        }


class ContingencyTableGenerator:
    """Generate and validate contingency tables"""
    
    def __init__(self):
        self.categorization_engine = CategorizationEngine()
    
    def generate_table(self, df: pd.DataFrame, config: ContingencyTableConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate contingency table from data and configuration"""
        
        # Extract variables
        row_data = df[config.row_variable].copy()
        col_data = df[config.column_variable].copy()
        
        # Apply categorization if needed
        if config.row_variable in df.columns and df[config.row_variable].dtype in ['float64', 'int64']:
            row_profile = self._analyze_variable(row_data)
            if row_profile['is_continuous'] and config.bin_continuous:
                row_data = self.categorization_engine.categorize_variable(
                    row_data, method=config.categorization_method, 
                    num_bins=config.num_bins, custom_bins=config.custom_bins
                )
        
        if config.column_variable in df.columns and df[config.column_variable].dtype in ['float64', 'int64']:
            col_profile = self._analyze_variable(col_data)
            if col_profile['is_continuous'] and config.bin_continuous:
                col_data = self.categorization_engine.categorize_variable(
                    col_data, method=config.categorization_method,
                    num_bins=config.num_bins, custom_bins=config.custom_bins
                )
        
        # Create contingency table
        contingency_table = pd.crosstab(
            row_data, 
            col_data, 
            margins=True, 
            margins_name='Total',
            dropna=False
        )
        
        # Generate validation statistics
        validation_stats = self._validate_table(contingency_table)
        
        return contingency_table, validation_stats
    
    def _analyze_variable(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze a variable to determine if it needs categorization"""
        unique_count = series.nunique()
        total_count = len(series)
        
        return {
            'is_continuous': pd.api.types.is_numeric_dtype(series) and unique_count > 20,
            'unique_count': unique_count,
            'cardinality_ratio': unique_count / total_count if total_count > 0 else 0
        }
    
    def _validate_table(self, contingency_table: pd.DataFrame) -> Dict[str, Any]:
        """Validate contingency table and compute statistics"""
        
        # Remove margins for statistical tests
        table_no_margins = contingency_table.iloc[:-1, :-1]
        
        if table_no_margins.empty or table_no_margins.sum().sum() == 0:
            return {
                'valid': False,
                'error': 'Empty contingency table',
                'chi2_stat': None,
                'p_value': None,
                'degrees_freedom': None,
                'expected_frequencies': None
            }
        
        try:
            # Chi-square test
            chi2_stat, p_value, dof, expected = chi2_contingency(table_no_margins)
            
            # Check assumptions
            min_expected = expected.min()
            cells_below_5 = (expected < 5).sum()
            total_cells = expected.size
            
            assumptions_met = min_expected >= 1 and cells_below_5 / total_cells <= 0.2
            
            return {
                'valid': True,
                'chi2_stat': chi2_stat,
                'p_value': p_value,
                'degrees_freedom': dof,
                'expected_frequencies': expected,
                'min_expected_frequency': min_expected,
                'cells_below_5': cells_below_5,
                'total_cells': total_cells,
                'assumptions_met': assumptions_met,
                'sample_size': table_no_margins.sum().sum()
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'chi2_stat': None,
                'p_value': None,
                'degrees_freedom': None,
                'expected_frequencies': None
            }


class CONFIRMDataConverter:
    """Main application class for the CONFIRM Data Converter"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CONFIRM Data Converter")
        self.root.geometry("1200x800")
        
        # Initialize modules
        self.data_import = DataImportModule()
        self.contingency_generator = ContingencyTableGenerator()
        self.quality_assessor = DataQualityAssessment()
        
        # Data storage
        self.current_data: Optional[pd.DataFrame] = None
        self.data_profiles: Dict[str, DataProfile] = {}
        self.current_contingency_table: Optional[pd.DataFrame] = None
        self.validation_stats: Dict[str, Any] = {}
        
        self._create_gui()
        
    def _create_gui(self):
        """Create the main GUI interface"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Data Import
        self.import_frame = ttk.Frame(notebook)
        notebook.add(self.import_frame, text="Data Import")
        self._create_import_tab()
        
        # Tab 2: Variable Selection
        self.selection_frame = ttk.Frame(notebook)
        notebook.add(self.selection_frame, text="Variable Selection")
        self.variable_interface = VariableSelectionInterface(self.selection_frame)
        
        # Tab 3: Data Quality
        self.quality_frame = ttk.Frame(notebook)
        notebook.add(self.quality_frame, text="Data Quality")
        self._create_quality_tab()
        
        # Tab 4: Contingency Table
        self.table_frame = ttk.Frame(notebook)
        notebook.add(self.table_frame, text="Contingency Table")
        self._create_table_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to import data")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _create_import_tab(self):
        """Create the data import tab"""
        # File selection frame
        file_frame = ttk.LabelFrame(self.import_frame, text="File Selection", padding="10")
        file_frame.pack(fill='x', padx=10, pady=5)
        
        # File path entry
        ttk.Label(file_frame, text="File Path:").grid(row=0, column=0, sticky='w', padx=(0, 10))
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=60)
        file_entry.grid(row=0, column=1, padx=(0, 10))
        
        ttk.Button(file_frame, text="Browse", command=self._browse_file).grid(row=0, column=2)
        ttk.Button(file_frame, text="Import", command=self._import_data).grid(row=0, column=3, padx=(10, 0))
        
        # Database connection frame
        db_frame = ttk.LabelFrame(self.import_frame, text="Database Connection (Optional)", padding="10")
        db_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(db_frame, text="SQL Query:").grid(row=0, column=0, sticky='nw', padx=(0, 10))
        self.sql_query_text = scrolledtext.ScrolledText(db_frame, height=3, width=80)
        self.sql_query_text.grid(row=0, column=1, columnspan=2)
        
        # Data preview frame
        preview_frame = ttk.LabelFrame(self.import_frame, text="Data Preview", padding="10")
        preview_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create treeview for data preview
        self.preview_tree = ttk.Treeview(preview_frame, show='headings', height=15)
        
        # Scrollbars for preview
        v_scrollbar = ttk.Scrollbar(preview_frame, orient='vertical', command=self.preview_tree.yview)
        h_scrollbar = ttk.Scrollbar(preview_frame, orient='horizontal', command=self.preview_tree.xview)
        self.preview_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.preview_tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)
    
    def _create_quality_tab(self):
        """Create the data quality assessment tab"""
        # Assessment button frame
        button_frame = ttk.Frame(self.quality_frame)
        button_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(button_frame, text="Assess Data Quality", command=self._assess_data_quality).pack(side='left')
        ttk.Button(button_frame, text="Export Quality Report", command=self._export_quality_report).pack(side='left', padx=(10, 0))
        
        # Quality report display
        report_frame = ttk.LabelFrame(self.quality_frame, text="Data Quality Report", padding="10")
        report_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.quality_text = scrolledtext.ScrolledText(report_frame, wrap=tk.WORD, font=('Courier', 10))
        self.quality_text.pack(fill='both', expand=True)
    
    def _create_table_tab(self):
        """Create the contingency table tab"""
        # Configuration frame
        config_frame = ttk.LabelFrame(self.table_frame, text="Table Configuration", padding="10")
        config_frame.pack(fill='x', padx=10, pady=5)
        
        # Categorization options
        ttk.Label(config_frame, text="Categorization Method:").grid(row=0, column=0, sticky='w', padx=(0, 10))
        self.cat_method_var = tk.StringVar(value='equal_width')
        cat_combo = ttk.Combobox(config_frame, textvariable=self.cat_method_var, 
                                values=['equal_width', 'equal_frequency', 'quantile', 'kmeans', 'jenks', 'custom'], 
                                state='readonly')
        cat_combo.grid(row=0, column=1, padx=(0, 20))
        
        ttk.Label(config_frame, text="Number of Bins:").grid(row=0, column=2, sticky='w', padx=(0, 10))
        self.num_bins_var = tk.StringVar(value='5')
        bins_spinbox = ttk.Spinbox(config_frame, from_=2, to=20, textvariable=self.num_bins_var, width=10)
        bins_spinbox.grid(row=0, column=3, padx=(0, 20))
        
        ttk.Button(config_frame, text="Generate Table", command=self._generate_contingency_table).grid(row=0, column=4)
        ttk.Button(config_frame, text="Custom Bins", command=self._open_custom_bins_dialog).grid(row=0, column=5, padx=(10, 0))
        
        # Results frame
        results_frame = ttk.LabelFrame(self.table_frame, text="Contingency Table Results", padding="10")
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create notebook for table and statistics
        results_notebook = ttk.Notebook(results_frame)
        results_notebook.pack(fill='both', expand=True)
        
        # Table display
        table_frame = ttk.Frame(results_notebook)
        results_notebook.add(table_frame, text="Contingency Table")
        
        self.table_tree = ttk.Treeview(table_frame, show='headings')
        table_scrollbar_v = ttk.Scrollbar(table_frame, orient='vertical', command=self.table_tree.yview)
        table_scrollbar_h = ttk.Scrollbar(table_frame, orient='horizontal', command=self.table_tree.xview)
        self.table_tree.configure(yscrollcommand=table_scrollbar_v.set, xscrollcommand=table_scrollbar_h.set)
        
        self.table_tree.grid(row=0, column=0, sticky='nsew')
        table_scrollbar_v.grid(row=0, column=1, sticky='ns')
        table_scrollbar_h.grid(row=1, column=0, sticky='ew')
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Visualization display
        viz_frame = ttk.Frame(results_notebook)
        results_notebook.add(viz_frame, text="Visualization")
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Statistics display
        stats_frame = ttk.Frame(results_notebook)
        results_notebook.add(stats_frame, text="Validation Statistics")
        
        self.stats_text = scrolledtext.ScrolledText(stats_frame, wrap=tk.WORD)
        self.stats_text.pack(fill='both', expand=True)
        
        # Export frame
        export_frame = ttk.Frame(self.table_frame)
        export_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(export_frame, text="Export to Excel", command=self._export_excel).pack(side='left', padx=(0, 10))
        ttk.Button(export_frame, text="Export to CSV", command=self._export_csv).pack(side='left', padx=(0, 10))
        ttk.Button(export_frame, text="Generate Report", command=self._generate_report).pack(side='left')
    
    def _browse_file(self):
        """Browse for data file"""
        filetypes = [
            ('All Supported', '*.csv;*.xlsx;*.xls;*.json;*.db;*.sqlite;*.sqlite3'),
            ('CSV files', '*.csv'),
            ('Excel files', '*.xlsx;*.xls'),
            ('JSON files', '*.json'),
            ('Database files', '*.db;*.sqlite;*.sqlite3'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=filetypes
        )
        
        if filename:
            self.file_path_var.set(filename)
    
    def _import_data(self):
        """Import data from selected file"""
        file_path = self.file_path_var.get()
        
        if not file_path:
            messagebox.showerror("Error", "Please select a file to import")
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("Error", "Selected file does not exist")
            return
        
        try:
            self.status_var.set("Importing data...")
            
            # Get SQL query if database file
            sql_query = None
            if file_path.lower().endswith(('.db', '.sqlite', '.sqlite3')):
                sql_query = self.sql_query_text.get(1.0, tk.END).strip()
                if sql_query:
                    self.current_data = self.data_import.import_data(file_path, query=sql_query)
                else:
                    self.current_data = self.data_import.import_data(file_path)
            else:
                self.current_data = self.data_import.import_data(file_path)
            
            # Generate data profiles
            self.data_profiles = self.data_import.detect_schema(self.current_data)
            
            # Update preview
            self._update_preview()
            
            # Update variable selection interface
            self.variable_interface.update_data_profiles(self.data_profiles)
            
            # Run automatic quality assessment
            self.quality_assessment = self.quality_assessor.assess_dataset(self.current_data)
            
            self.status_var.set(f"Data imported successfully: {len(self.current_data)} rows, {len(self.current_data.columns)} columns")
            
        except Exception as e:
            error_msg = f"Error importing data: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("Import Error", error_msg)
            self.status_var.set("Import failed")
    
    def _update_preview(self):
        """Update the data preview display"""
        if self.current_data is None:
            return
        
        # Clear existing columns and data
        self.preview_tree.delete(*self.preview_tree.get_children())
        
        # Get preview data
        preview_data = self.data_import.get_preview(self.current_data, 100)
        
        # Set up columns
        columns = list(preview_data.columns)
        self.preview_tree['columns'] = columns
        
        for col in columns:
            self.preview_tree.heading(col, text=col)
            self.preview_tree.column(col, width=120)
        
        # Insert data
        for index, row in preview_data.iterrows():
            values = [str(val) if pd.notna(val) else '' for val in row]
            self.preview_tree.insert('', 'end', values=values)
    
    def _generate_contingency_table(self):
        """Generate contingency table based on current configuration"""
        if self.current_data is None:
            messagebox.showerror("Error", "Please import data first")
            return
        
        config = self.variable_interface.get_selection_config()
        if config is None:
            messagebox.showerror("Error", "Please select both row and column variables")
            return
        
        try:
            self.status_var.set("Generating contingency table...")
            
            # Update config with categorization settings
            config.bin_continuous = True
            config.num_bins = int(self.num_bins_var.get())
            config.categorization_method = self.cat_method_var.get()
            
            # Generate table
            self.current_contingency_table, self.validation_stats = self.contingency_generator.generate_table(
                self.current_data, config
            )
            
            # Update displays
            self._update_table_display()
            self._update_stats_display()
            self._update_visualization()
            
            self.status_var.set("Contingency table generated successfully")
            
        except Exception as e:
            error_msg = f"Error generating contingency table: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("Generation Error", error_msg)
            self.status_var.set("Table generation failed")
    
    def _update_table_display(self):
        """Update the contingency table display"""
        if self.current_contingency_table is None:
            return
        
        # Clear existing data
        self.table_tree.delete(*self.table_tree.get_children())
        
        # Set up columns
        columns = [''] + list(self.current_contingency_table.columns)
        self.table_tree['columns'] = columns
        
        for col in columns:
            self.table_tree.heading(col, text=col)
            self.table_tree.column(col, width=100)
        
        # Insert data
        for index, row in self.current_contingency_table.iterrows():
            values = [str(index)] + [str(val) for val in row]
            self.table_tree.insert('', 'end', values=values)
    
    def _update_stats_display(self):
        """Update the validation statistics display"""
        if not self.validation_stats:
            return
        
        stats_text = []
        
        if self.validation_stats.get('valid', False):
            stats_text.append("CONTINGENCY TABLE VALIDATION RESULTS")
            stats_text.append("=" * 50)
            stats_text.append("")
            
            # Basic statistics
            stats_text.append(f"Sample Size: {self.validation_stats['sample_size']:,}")
            stats_text.append(f"Degrees of Freedom: {self.validation_stats['degrees_freedom']}")
            stats_text.append("")
            
            # Chi-square test results
            stats_text.append("CHI-SQUARE TEST RESULTS:")
            stats_text.append(f"Chi-square statistic: {self.validation_stats['chi2_stat']:.4f}")
            stats_text.append(f"P-value: {self.validation_stats['p_value']:.6f}")
            
            if self.validation_stats['p_value'] < 0.05:
                stats_text.append("Result: Statistically significant association (p < 0.05)")
            else:
                stats_text.append("Result: No statistically significant association (p ≥ 0.05)")
            
            stats_text.append("")
            
            # Assumption checking
            stats_text.append("ASSUMPTION VALIDATION:")
            stats_text.append(f"Minimum expected frequency: {self.validation_stats['min_expected_frequency']:.2f}")
            stats_text.append(f"Cells with expected frequency < 5: {self.validation_stats['cells_below_5']}/{self.validation_stats['total_cells']}")
            
            if self.validation_stats['assumptions_met']:
                stats_text.append("✓ Chi-square test assumptions are satisfied")
            else:
                stats_text.append("⚠ Chi-square test assumptions may be violated")
                stats_text.append("Consider combining categories or collecting more data")
            
        else:
            stats_text.append("VALIDATION ERROR")
            stats_text.append("=" * 30)
            stats_text.append(f"Error: {self.validation_stats.get('error', 'Unknown error')}")
        
        # Update display
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, '\n'.join(stats_text))
    
    def _update_visualization(self):
        """Update the contingency table visualization"""
        if self.current_contingency_table is None:
            return
        
        # Clear previous plots
        self.fig.clear()
        
        # Remove margins for visualization
        table_no_margins = self.current_contingency_table.iloc[:-1, :-1]
        
        if table_no_margins.empty:
            return
        
        # Create subplots
        ax1 = self.fig.add_subplot(221)  # Heatmap
        ax2 = self.fig.add_subplot(222)  # Bar chart
        ax3 = self.fig.add_subplot(223)  # Stacked bar
        ax4 = self.fig.add_subplot(224)  # Mosaic-style
        
        # Heatmap
        sns.heatmap(table_no_margins, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Contingency Table Heatmap')
        ax1.set_xlabel(self.variable_interface.selected_col_var.get())
        ax1.set_ylabel(self.variable_interface.selected_row_var.get())
        
        # Bar chart of totals
        row_totals = table_no_margins.sum(axis=1)
        row_totals.plot(kind='bar', ax=ax2, color='skyblue')
        ax2.set_title('Row Variable Distribution')
        ax2.set_xlabel(self.variable_interface.selected_row_var.get())
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        # Stacked bar chart
        table_no_margins.plot(kind='bar', stacked=True, ax=ax3, colormap='Set3')
        ax3.set_title('Stacked Distribution')
        ax3.set_xlabel(self.variable_interface.selected_row_var.get())
        ax3.set_ylabel('Count')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(title=self.variable_interface.selected_col_var.get(), bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Normalized heatmap (percentages)
        table_normalized = table_no_margins.div(table_no_margins.sum().sum()) * 100
        sns.heatmap(table_normalized, annot=True, fmt='.1f', cmap='Oranges', ax=ax4)
        ax4.set_title('Percentage Distribution')
        ax4.set_xlabel(self.variable_interface.selected_col_var.get())
        ax4.set_ylabel(self.variable_interface.selected_row_var.get())
        
        # Adjust layout
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _export_excel(self):
        """Export contingency table to Excel"""
        if self.current_contingency_table is None:
            messagebox.showerror("Error", "No contingency table to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Contingency Table",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    self.current_contingency_table.to_excel(writer, sheet_name='Contingency_Table')
                    
                    # Add validation statistics if available
                    if self.validation_stats and self.validation_stats.get('valid', False):
                        stats_df = pd.DataFrame([
                            ['Chi-square statistic', self.validation_stats['chi2_stat']],
                            ['P-value', self.validation_stats['p_value']],
                            ['Degrees of freedom', self.validation_stats['degrees_freedom']],
                            ['Sample size', self.validation_stats['sample_size']],
                            ['Assumptions met', self.validation_stats['assumptions_met']]
                        ], columns=['Statistic', 'Value'])
                        
                        stats_df.to_excel(writer, sheet_name='Validation_Statistics', index=False)
                
                messagebox.showinfo("Success", f"Contingency table exported to {filename}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting file: {str(e)}")
    
    def _export_csv(self):
        """Export contingency table to CSV"""
        if self.current_contingency_table is None:
            messagebox.showerror("Error", "No contingency table to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Contingency Table",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.current_contingency_table.to_csv(filename)
                messagebox.showinfo("Success", f"Contingency table exported to {filename}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting file: {str(e)}")
    
    def _generate_report(self):
        """Generate comprehensive analysis report"""
        if self.current_contingency_table is None:
            messagebox.showerror("Error", "No contingency table to report")
            return
        
        # This would integrate with the existing report generation system
        messagebox.showinfo("Report Generation", "Report generation feature will be integrated with existing RADAR reporting system")
    
    def _open_custom_bins_dialog(self):
        """Open dialog for custom bin configuration"""
        if self.current_data is None:
            messagebox.showerror("Error", "Please import data first")
            return
        
        config = self.variable_interface.get_selection_config()
        if config is None:
            messagebox.showerror("Error", "Please select variables first")
            return
        
        # Create custom bins dialog
        dialog = CustomBinsDialog(self.root, self.current_data, config)
        self.root.wait_window(dialog.dialog)
        
        if dialog.result:
            # Update configuration with custom bins
            self.custom_bins = dialog.result
            messagebox.showinfo("Success", f"Custom bins configured: {len(self.custom_bins)-1} bins")
    
    def _assess_data_quality(self):
        """Perform data quality assessment"""
        if self.current_data is None:
            messagebox.showerror("Error", "Please import data first")
            return
        
        try:
            self.status_var.set("Assessing data quality...")
            
            # Perform assessment
            self.quality_assessment = self.quality_assessor.assess_dataset(self.current_data)
            
            # Generate and display report
            quality_report = self.quality_assessor.generate_quality_report(self.quality_assessment)
            
            self.quality_text.delete(1.0, tk.END)
            self.quality_text.insert(1.0, quality_report)
            
            self.status_var.set("Data quality assessment completed")
            
        except Exception as e:
            error_msg = f"Error assessing data quality: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("Assessment Error", error_msg)
            self.status_var.set("Quality assessment failed")
    
    def _export_quality_report(self):
        """Export data quality report"""
        if not hasattr(self, 'quality_assessment'):
            messagebox.showerror("Error", "Please run data quality assessment first")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Quality Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                quality_report = self.quality_assessor.generate_quality_report(self.quality_assessment)
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(quality_report)
                
                messagebox.showinfo("Success", f"Quality report exported to {filename}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting report: {str(e)}")


class CustomBinsDialog:
    """Dialog for configuring custom bins for continuous variables"""
    
    def __init__(self, parent, data: pd.DataFrame, config: ContingencyTableConfig):
        self.parent = parent
        self.data = data
        self.config = config
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Custom Bins Configuration")
        self.dialog.geometry("600x500")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self._create_dialog()
        
        # Center dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")
    
    def _create_dialog(self):
        """Create the custom bins dialog interface"""
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Variable selection
        var_frame = ttk.LabelFrame(main_frame, text="Variable Information", padding="10")
        var_frame.pack(fill='x', pady=(0, 10))
        
        # Check which variables are continuous
        continuous_vars = []
        for var in [self.config.row_variable, self.config.column_variable]:
            if var in self.data.columns and pd.api.types.is_numeric_dtype(self.data[var]):
                continuous_vars.append(var)
        
        if not continuous_vars:
            ttk.Label(var_frame, text="No continuous variables selected for binning").pack()
            ttk.Button(main_frame, text="Close", command=self.dialog.destroy).pack(pady=10)
            return
        
        # Variable selection
        ttk.Label(var_frame, text="Select variable to configure:").pack(anchor='w')
        self.selected_var = tk.StringVar(value=continuous_vars[0])
        var_combo = ttk.Combobox(var_frame, textvariable=self.selected_var, 
                                values=continuous_vars, state='readonly')
        var_combo.pack(fill='x', pady=5)
        var_combo.bind('<<ComboboxSelected>>', self._on_variable_changed)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(main_frame, text="Variable Statistics", padding="10")
        stats_frame.pack(fill='x', pady=(0, 10))
        
        self.stats_text = tk.Text(stats_frame, height=6, wrap=tk.WORD)
        self.stats_text.pack(fill='x')
        
        # Binning method frame
        method_frame = ttk.LabelFrame(main_frame, text="Binning Configuration", padding="10")
        method_frame.pack(fill='x', pady=(0, 10))
        
        # Method selection
        ttk.Label(method_frame, text="Binning Method:").grid(row=0, column=0, sticky='w', padx=(0, 10))
        self.method_var = tk.StringVar(value='equal_width')
        method_combo = ttk.Combobox(method_frame, textvariable=self.method_var,
                                   values=['equal_width', 'equal_frequency', 'quantile', 'kmeans', 'custom'],
                                   state='readonly')
        method_combo.grid(row=0, column=1, sticky='ew', padx=(0, 20))
        method_combo.bind('<<ComboboxSelected>>', self._on_method_changed)
        
        # Number of bins
        ttk.Label(method_frame, text="Number of Bins:").grid(row=0, column=2, sticky='w', padx=(0, 10))
        self.bins_var = tk.StringVar(value='5')
        bins_spinbox = ttk.Spinbox(method_frame, from_=2, to=20, textvariable=self.bins_var)
        bins_spinbox.grid(row=0, column=3)
        
        method_frame.grid_columnconfigure(1, weight=1)
        
        # Custom bins entry (initially hidden)
        self.custom_frame = ttk.Frame(method_frame)
        self.custom_frame.grid(row=1, column=0, columnspan=4, sticky='ew', pady=(10, 0))
        self.custom_frame.grid_remove()
        
        ttk.Label(self.custom_frame, text="Custom Bin Edges (comma-separated):").pack(anchor='w')
        self.custom_bins_var = tk.StringVar()
        custom_entry = ttk.Entry(self.custom_frame, textvariable=self.custom_bins_var, width=50)
        custom_entry.pack(fill='x', pady=5)
        
        # Preview frame
        preview_frame = ttk.LabelFrame(main_frame, text="Binning Preview", padding="10")
        preview_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        self.preview_text = scrolledtext.ScrolledText(preview_frame, height=8)
        self.preview_text.pack(fill='both', expand=True)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x')
        
        ttk.Button(button_frame, text="Preview Bins", command=self._preview_bins).pack(side='left', padx=(0, 10))
        ttk.Button(button_frame, text="Apply", command=self._apply_bins).pack(side='left', padx=(0, 10))
        ttk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side='right')
        
        # Initialize display
        self._on_variable_changed()
    
    def _on_variable_changed(self, event=None):
        """Handle variable selection change"""
        var_name = self.selected_var.get()
        if var_name not in self.data.columns:
            return
        
        series = self.data[var_name]
        
        # Update statistics
        stats = [
            f"Variable: {var_name}",
            f"Data Type: {series.dtype}",
            f"Total Values: {len(series):,}",
            f"Unique Values: {series.nunique():,}",
            f"Null Values: {series.isnull().sum():,}",
            f"Range: {series.min():.2f} - {series.max():.2f}",
            f"Mean: {series.mean():.2f}",
            f"Standard Deviation: {series.std():.2f}"
        ]
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, '\n'.join(stats))
        
        # Update custom bins placeholder
        min_val, max_val = series.min(), series.max()
        suggested_bins = np.linspace(min_val, max_val, 6)
        self.custom_bins_var.set(', '.join([f"{x:.2f}" for x in suggested_bins]))
    
    def _on_method_changed(self, event=None):
        """Handle binning method change"""
        method = self.method_var.get()
        
        if method == 'custom':
            self.custom_frame.grid()
        else:
            self.custom_frame.grid_remove()
    
    def _preview_bins(self):
        """Preview the binning results"""
        var_name = self.selected_var.get()
        method = self.method_var.get()
        num_bins = int(self.bins_var.get())
        
        try:
            from confirm_data_converter import CategorizationEngine
            engine = CategorizationEngine()
            
            series = self.data[var_name].dropna()
            
            if method == 'custom':
                custom_bins_str = self.custom_bins_var.get()
                custom_bins = [float(x.strip()) for x in custom_bins_str.split(',')]
                categorized = engine.categorize_variable(series, method=method, custom_bins=custom_bins)
            else:
                categorized = engine.categorize_variable(series, method=method, num_bins=num_bins)
            
            # Generate preview
            value_counts = categorized.value_counts().sort_index()
            
            preview_lines = [
                f"Binning Preview for {var_name}",
                f"Method: {method}",
                f"Number of bins: {len(value_counts)}",
                "",
                "Bin Distribution:"
            ]
            
            for bin_label, count in value_counts.items():
                percentage = (count / len(categorized)) * 100
                preview_lines.append(f"  {bin_label}: {count:,} ({percentage:.1f}%)")
            
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(1.0, '\n'.join(preview_lines))
            
        except Exception as e:
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(1.0, f"Error previewing bins: {str(e)}")
    
    def _apply_bins(self):
        """Apply the custom bins configuration"""
        try:
            method = self.method_var.get()
            
            if method == 'custom':
                custom_bins_str = self.custom_bins_var.get()
                self.result = [float(x.strip()) for x in custom_bins_str.split(',')]
            else:
                self.result = {
                    'method': method,
                    'num_bins': int(self.bins_var.get())
                }
            
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Invalid bin configuration: {str(e)}")


class DataQualityAssessment:
    """Assess data quality and provide recommendations for contingency table analysis"""
    
    def __init__(self):
        pass
    
    def assess_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality assessment"""
        assessment = {
            'overall_score': 0,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'variable_assessments': {},
            'recommendations': [],
            'warnings': [],
            'suitability_score': 0
        }
        
        # Assess each variable
        suitable_vars = 0
        total_quality_score = 0
        
        for col in df.columns:
            var_assessment = self._assess_variable(df[col], col)
            assessment['variable_assessments'][col] = var_assessment
            
            if var_assessment['suitable_for_contingency']:
                suitable_vars += 1
            
            total_quality_score += var_assessment['quality_score']
        
        # Calculate overall scores
        assessment['overall_score'] = total_quality_score / len(df.columns) if len(df.columns) > 0 else 0
        assessment['suitability_score'] = (suitable_vars / len(df.columns)) * 100 if len(df.columns) > 0 else 0
        
        # Generate recommendations
        assessment['recommendations'] = self._generate_recommendations(assessment)
        assessment['warnings'] = self._generate_warnings(assessment)
        
        return assessment
    
    def _assess_variable(self, series: pd.Series, name: str) -> Dict[str, Any]:
        """Assess individual variable quality"""
        assessment = {
            'name': name,
            'data_type': str(series.dtype),
            'total_count': len(series),
            'unique_count': series.nunique(),
            'null_count': series.isnull().sum(),
            'null_percentage': (series.isnull().sum() / len(series)) * 100,
            'quality_score': 0,
            'suitable_for_contingency': False,
            'issues': [],
            'recommendations': []
        }
        
        # Calculate quality score (0-100)
        quality_factors = []
        
        # Null percentage factor (lower is better)
        null_factor = max(0, 100 - assessment['null_percentage'] * 2)
        quality_factors.append(null_factor)
        
        # Uniqueness factor (depends on variable type)
        cardinality_ratio = assessment['unique_count'] / assessment['total_count']
        
        if pd.api.types.is_numeric_dtype(series):
            # For numeric variables
            assessment['is_continuous'] = True
            
            # Check for distribution properties
            non_null_series = series.dropna()
            if len(non_null_series) > 3:
                try:
                    assessment['skewness'] = skew(non_null_series)
                    assessment['kurtosis'] = kurtosis(non_null_series)
                    
                    # Normality test
                    if len(non_null_series) >= 8:
                        stat, p_val = normaltest(non_null_series)
                        assessment['normality_p_value'] = p_val
                        assessment['is_normal'] = p_val > 0.05
                except:
                    pass
            
            # Continuous variables need binning for contingency tables
            if cardinality_ratio > 0.1:  # High cardinality
                quality_factors.append(70)  # Good for binning
                assessment['suitable_for_contingency'] = True
                assessment['recommendations'].append("Suitable for binning into categories")
            else:
                quality_factors.append(85)  # Already discrete
                assessment['suitable_for_contingency'] = True
                assessment['recommendations'].append("Already discrete, good for contingency analysis")
        
        else:
            # For categorical variables
            assessment['is_continuous'] = False
            
            if 2 <= assessment['unique_count'] <= 50:
                quality_factors.append(90)  # Ideal for contingency tables
                assessment['suitable_for_contingency'] = True
                assessment['recommendations'].append("Ideal for contingency table analysis")
            elif assessment['unique_count'] < 2:
                quality_factors.append(20)  # Not useful
                assessment['issues'].append("Only one category - not suitable for analysis")
            elif assessment['unique_count'] > 50:
                quality_factors.append(60)  # Too many categories
                assessment['issues'].append("Too many categories - consider grouping")
                assessment['recommendations'].append("Consider combining similar categories")
            else:
                quality_factors.append(80)
        
        # Data completeness factor
        if assessment['null_percentage'] == 0:
            quality_factors.append(100)
        elif assessment['null_percentage'] < 5:
            quality_factors.append(90)
        elif assessment['null_percentage'] < 20:
            quality_factors.append(70)
            assessment['issues'].append("Moderate missing data")
        else:
            quality_factors.append(40)
            assessment['issues'].append("High missing data percentage")
        
        # Calculate final quality score
        assessment['quality_score'] = sum(quality_factors) / len(quality_factors)
        
        return assessment
    
    def _generate_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations for the dataset"""
        recommendations = []
        
        suitable_vars = [v for v in assessment['variable_assessments'].values() 
                        if v['suitable_for_contingency']]
        
        if len(suitable_vars) >= 2:
            recommendations.append(f"✓ Dataset has {len(suitable_vars)} variables suitable for contingency analysis")
        else:
            recommendations.append("⚠ Dataset has fewer than 2 suitable variables for contingency analysis")
        
        # Sample size recommendations
        if assessment['total_rows'] < 30:
            recommendations.append("⚠ Small sample size - consider collecting more data for robust analysis")
        elif assessment['total_rows'] < 100:
            recommendations.append("• Adequate sample size for basic analysis")
        else:
            recommendations.append("✓ Good sample size for reliable statistical analysis")
        
        # Overall quality recommendations
        if assessment['overall_score'] >= 80:
            recommendations.append("✓ High data quality - dataset is well-suited for analysis")
        elif assessment['overall_score'] >= 60:
            recommendations.append("• Moderate data quality - some improvements possible")
        else:
            recommendations.append("⚠ Low data quality - consider data cleaning before analysis")
        
        return recommendations
    
    def _generate_warnings(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate warnings for potential issues"""
        warnings = []
        
        # Check for variables with high missing data
        high_missing_vars = [v['name'] for v in assessment['variable_assessments'].values() 
                           if v['null_percentage'] > 20]
        
        if high_missing_vars:
            warnings.append(f"High missing data in variables: {', '.join(high_missing_vars)}")
        
        # Check for variables with too many categories
        high_cardinality_vars = [v['name'] for v in assessment['variable_assessments'].values() 
                               if not v['is_continuous'] and v['unique_count'] > 20]
        
        if high_cardinality_vars:
            warnings.append(f"High cardinality categorical variables: {', '.join(high_cardinality_vars)}")
        
        # Check sample size vs. variable complexity
        total_categories = sum(v['unique_count'] for v in assessment['variable_assessments'].values() 
                             if v['suitable_for_contingency'])
        
        if assessment['total_rows'] < total_categories * 5:
            warnings.append("Sample size may be too small relative to the number of categories")
        
        return warnings
    
    def generate_quality_report(self, assessment: Dict[str, Any]) -> str:
        """Generate a formatted quality assessment report"""
        report_lines = [
            "DATA QUALITY ASSESSMENT REPORT",
            "=" * 50,
            "",
            f"Dataset Overview:",
            f"  • Total Rows: {assessment['total_rows']:,}",
            f"  • Total Columns: {assessment['total_columns']}",
            f"  • Overall Quality Score: {assessment['overall_score']:.1f}/100",
            f"  • Contingency Analysis Suitability: {assessment['suitability_score']:.1f}%",
            "",
            "Variable Assessment:",
        ]
        
        # Add variable details
        for var_name, var_assessment in assessment['variable_assessments'].items():
            status = "✓" if var_assessment['suitable_for_contingency'] else "✗"
            report_lines.append(
                f"  {status} {var_name}: {var_assessment['quality_score']:.1f}/100 "
                f"({var_assessment['unique_count']} unique, {var_assessment['null_percentage']:.1f}% missing)"
            )
        
        # Add recommendations
        if assessment['recommendations']:
            report_lines.extend(["", "Recommendations:"])
            for rec in assessment['recommendations']:
                report_lines.append(f"  {rec}")
        
        # Add warnings
        if assessment['warnings']:
            report_lines.extend(["", "Warnings:"])
            for warning in assessment['warnings']:
                report_lines.append(f"  ⚠ {warning}")
        
        return '\n'.join(report_lines)
    
    def run(self):
        """Run the application"""
        try:
            self.root.mainloop()
        except Exception as e:
            logger.error(f"Application error: {str(e)}")
            messagebox.showerror("Application Error", str(e))


def main():
    """Main entry point"""
    try:
        app = CONFIRMDataConverter()
        app.run()
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()