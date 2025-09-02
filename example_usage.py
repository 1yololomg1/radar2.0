#!/usr/bin/env python3
"""
Example usage of CONFIRM Data Converter components
Demonstrates programmatic usage without GUI
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from confirm_data_converter import (
    DataImportModule, 
    CategorizationEngine, 
    ContingencyTableGenerator,
    ContingencyTableConfig,
    DataQualityAssessment
)


def example_basic_usage():
    """Basic usage example"""
    print("CONFIRM Data Converter - Basic Usage Example")
    print("=" * 50)
    
    # 1. Import data
    importer = DataImportModule()
    df = importer.import_data('sample_data.csv')
    print(f"Imported data: {len(df)} rows, {len(df.columns)} columns")
    
    # 2. Assess data quality
    quality_assessor = DataQualityAssessment()
    assessment = quality_assessor.assess_dataset(df)
    print(f"Data quality score: {assessment['overall_score']:.1f}/100")
    
    # 3. Generate contingency table
    config = ContingencyTableConfig(
        row_variable='education_level',
        column_variable='purchase_category'
    )
    
    generator = ContingencyTableGenerator()
    table, stats = generator.generate_table(df, config)
    
    print(f"\nContingency Table:")
    print(table)
    
    print(f"\nStatistical Results:")
    print(f"Chi-square statistic: {stats['chi2_stat']:.4f}")
    print(f"P-value: {stats['p_value']:.6f}")
    print(f"Degrees of freedom: {stats['degrees_freedom']}")
    print(f"Assumptions met: {stats['assumptions_met']}")
    
    # 4. Export results
    table.to_csv('example_output.csv')
    print(f"\nResults exported to: example_output.csv")


def example_advanced_usage():
    """Advanced usage with continuous variable binning"""
    print("\n\nAdvanced Usage - Continuous Variable Binning")
    print("=" * 50)
    
    # Import data
    importer = DataImportModule()
    df = importer.import_data('sample_data.csv')
    
    # Configure for continuous variables
    config = ContingencyTableConfig(
        row_variable='age',
        column_variable='income',
        bin_continuous=True,
        num_bins=3,
        categorization_method='kmeans'  # Use K-means clustering
    )
    
    generator = ContingencyTableGenerator()
    table, stats = generator.generate_table(df, config)
    
    print(f"Continuous Variables Contingency Table (K-means binning):")
    print(table)
    
    print(f"\nStatistical Results:")
    print(f"Chi-square statistic: {stats['chi2_stat']:.4f}")
    print(f"P-value: {stats['p_value']:.6f}")
    
    # Test different binning methods
    print(f"\nTesting Different Binning Methods:")
    categorizer = CategorizationEngine()
    
    for method in ['equal_width', 'equal_frequency', 'quantile']:
        categorized = categorizer.categorize_variable(df['age'], method=method, num_bins=4)
        bins_created = categorized.nunique()
        print(f"  {method}: {bins_created} bins")


def example_json_usage():
    """Example with JSON survey data"""
    print("\n\nJSON Survey Data Example")
    print("=" * 50)
    
    # Import JSON data
    importer = DataImportModule()
    df = importer.import_data('sample_survey_data.json')
    
    print(f"Survey data: {len(df)} responses, {len(df.columns)} variables")
    
    # Analyze satisfaction vs product category
    config = ContingencyTableConfig(
        row_variable='satisfaction_rating',
        column_variable='product_category'
    )
    
    generator = ContingencyTableGenerator()
    table, stats = generator.generate_table(df, config)
    
    print(f"\nSatisfaction vs Product Category:")
    print(table)
    
    # Analyze with continuous variable (annual spending)
    config_continuous = ContingencyTableConfig(
        row_variable='annual_spending',
        column_variable='satisfaction_rating',
        bin_continuous=True,
        num_bins=3,
        categorization_method='quantile'
    )
    
    table_cont, stats_cont = generator.generate_table(df, config_continuous)
    
    print(f"\nSpending vs Satisfaction (with binning):")
    print(table_cont)


if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    example_advanced_usage()
    example_json_usage()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("For GUI interface, run: python3 launch_confirm_converter.py")