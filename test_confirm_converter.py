#!/usr/bin/env python3
"""
Test script for CONFIRM Data Converter
Validates functionality with sample data
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from confirm_data_converter import (
    DataImportModule, 
    CategorizationEngine, 
    ContingencyTableGenerator,
    ContingencyTableConfig
)


def test_data_import():
    """Test data import functionality"""
    print("Testing Data Import Module...")
    
    importer = DataImportModule()
    
    # Test CSV import
    try:
        df_csv = importer.import_data('sample_data.csv')
        print(f"✓ CSV import successful: {len(df_csv)} rows, {len(df_csv.columns)} columns")
        
        # Test schema detection
        profiles = importer.detect_schema(df_csv)
        print(f"✓ Schema detection: {len(profiles)} variables profiled")
        
        # Print some profiles
        for name, profile in list(profiles.items())[:3]:
            print(f"  - {name}: {profile.dtype}, {profile.unique_count} unique values, "
                  f"{'categorical' if profile.is_categorical else 'continuous'}")
        
    except Exception as e:
        print(f"✗ CSV import failed: {e}")
    
    # Test JSON import
    try:
        df_json = importer.import_data('sample_survey_data.json')
        print(f"✓ JSON import successful: {len(df_json)} rows, {len(df_json.columns)} columns")
        
    except Exception as e:
        print(f"✗ JSON import failed: {e}")
    
    return df_csv, profiles


def test_categorization():
    """Test categorization engine"""
    print("\nTesting Categorization Engine...")
    
    engine = CategorizationEngine()
    
    # Create test data
    np.random.seed(42)
    continuous_data = pd.Series(np.random.normal(50, 15, 1000))
    
    # Test different methods
    methods = ['equal_width', 'equal_frequency', 'quantile', 'kmeans']
    
    for method in methods:
        try:
            categorized = engine.categorize_variable(continuous_data, method=method, num_bins=5)
            unique_cats = categorized.nunique()
            print(f"✓ {method} categorization: {unique_cats} categories created")
        except Exception as e:
            print(f"✗ {method} categorization failed: {e}")
    
    # Test bin suggestions
    suggestions = engine.suggest_bins(continuous_data)
    print(f"✓ Bin suggestions: {suggestions['num_bins']} bins using {suggestions['method']} method")
    print(f"  Reason: {suggestions['reason']}")


def test_contingency_generation():
    """Test contingency table generation"""
    print("\nTesting Contingency Table Generation...")
    
    # Load sample data
    importer = DataImportModule()
    df = importer.import_data('sample_data.csv')
    
    generator = ContingencyTableGenerator()
    
    # Test with categorical variables
    config = ContingencyTableConfig(
        row_variable='education_level',
        column_variable='purchase_category'
    )
    
    try:
        table, stats = generator.generate_table(df, config)
        print(f"✓ Contingency table generated: {table.shape}")
        print(f"✓ Validation statistics computed: Chi2={stats.get('chi2_stat', 'N/A'):.4f}")
        print(f"  P-value: {stats.get('p_value', 'N/A'):.6f}")
        print(f"  Assumptions met: {stats.get('assumptions_met', 'N/A')}")
        
        print("\nContingency Table:")
        print(table)
        
    except Exception as e:
        print(f"✗ Contingency table generation failed: {e}")
    
    # Test with continuous variables
    config_continuous = ContingencyTableConfig(
        row_variable='age',
        column_variable='income',
        bin_continuous=True,
        num_bins=4,
        categorization_method='quantile'
    )
    
    try:
        table_cont, stats_cont = generator.generate_table(df, config_continuous)
        print(f"\n✓ Continuous variable contingency table: {table_cont.shape}")
        print(f"✓ Chi2={stats_cont.get('chi2_stat', 'N/A'):.4f}, P-value: {stats_cont.get('p_value', 'N/A'):.6f}")
        
    except Exception as e:
        print(f"✗ Continuous variable table generation failed: {e}")


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\nTesting Edge Cases...")
    
    # Test with missing data
    df_with_nulls = pd.DataFrame({
        'var1': [1, 2, np.nan, 4, 5, np.nan],
        'var2': ['A', 'B', 'A', np.nan, 'C', 'B']
    })
    
    generator = ContingencyTableGenerator()
    config = ContingencyTableConfig(
        row_variable='var1',
        column_variable='var2',
        bin_continuous=True,
        num_bins=3
    )
    
    try:
        table, stats = generator.generate_table(df_with_nulls, config)
        print(f"✓ Handled missing data: {table.shape}")
        
    except Exception as e:
        print(f"✗ Missing data handling failed: {e}")
    
    # Test with single category
    df_single = pd.DataFrame({
        'var1': ['A'] * 10,
        'var2': ['X', 'Y'] * 5
    })
    
    config_single = ContingencyTableConfig(
        row_variable='var1',
        column_variable='var2'
    )
    
    try:
        table_single, stats_single = generator.generate_table(df_single, config_single)
        print(f"✓ Handled single category: {table_single.shape}")
        
    except Exception as e:
        print(f"✗ Single category handling failed: {e}")


def main():
    """Run all tests"""
    print("CONFIRM Data Converter - Test Suite")
    print("=" * 50)
    
    # Test data import
    df, profiles = test_data_import()
    
    # Test categorization
    test_categorization()
    
    # Test contingency table generation
    test_contingency_generation()
    
    # Test edge cases
    test_edge_cases()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")


if __name__ == "__main__":
    main()