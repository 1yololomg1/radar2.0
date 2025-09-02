"""
Test script for CONFIRM Data Converter
Tests the core functionality of each module
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the workspace to path
sys.path.insert(0, '/workspace')

# Import our modules
from data_import import DataImporter
from schema_detector import SchemaDetector
from variable_selector import VariableSelector
from contingency_generator import ContingencyTableGenerator
from recommendation_engine import RecommendationEngine

def test_data_import():
    """Test data import functionality"""
    print("Testing Data Import Module...")
    
    # Create test data
    test_data = pd.DataFrame({
        'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'] * 10,
        'outcome': ['Success', 'Failure', 'Success', 'Success', 'Failure', 
                   'Success', 'Failure', 'Success', 'Failure', 'Success'] * 10,
        'value': np.random.randn(100),
        'id': range(100)
    })
    
    # Save as CSV
    test_data.to_csv('/tmp/test_data.csv', index=False)
    
    # Test import
    importer = DataImporter()
    imported_df = importer.import_csv('/tmp/test_data.csv')
    
    assert len(imported_df) == 100
    assert list(imported_df.columns) == ['category', 'outcome', 'value', 'id']
    print("✓ Data import successful")
    
    return imported_df

def test_schema_detection(df):
    """Test schema detection"""
    print("\nTesting Schema Detection Module...")
    
    detector = SchemaDetector()
    schema = detector.detect_schema(df)
    
    assert 'category' in schema
    assert schema['category']['data_type'] == 'categorical'
    assert schema['value']['data_type'] == 'numeric'
    
    print("✓ Schema detection successful")
    print(f"  - Detected {len(schema)} columns")
    print(f"  - Category column type: {schema['category']['data_type']}")
    print(f"  - Value column type: {schema['value']['data_type']}")
    
    return schema

def test_variable_selection(df, schema):
    """Test variable selection"""
    print("\nTesting Variable Selection Module...")
    
    selector = VariableSelector()
    
    # Analyze single variable
    analysis = selector.analyze_variable(df, 'category')
    assert analysis['unique_count'] == 3
    assert analysis['is_suitable'] == True
    
    # Analyze pair
    pair_analysis = selector.analyze_pair(df, 'category', 'outcome')
    assert pair_analysis['is_valid'] == True
    assert pair_analysis['table_shape'] == (3, 2)
    
    print("✓ Variable selection successful")
    print(f"  - Category unique values: {analysis['unique_count']}")
    print(f"  - Contingency table shape: {pair_analysis['table_shape']}")
    
    return selector

def test_contingency_generation(df):
    """Test contingency table generation"""
    print("\nTesting Contingency Table Generation Module...")
    
    generator = ContingencyTableGenerator()
    
    result = generator.generate(
        df, 
        'category', 
        'outcome',
        normalize=None,
        margins=True
    )
    
    assert result['counts'] is not None
    assert 'chi2' in result['statistics']
    assert 'p_value' in result['statistics']
    assert 'cramers_v' in result['statistics']
    
    print("✓ Contingency table generation successful")
    print(f"  - Chi-square: {result['statistics']['chi2']:.4f}")
    print(f"  - P-value: {result['statistics']['p_value']:.6f}")
    print(f"  - Cramér's V: {result['statistics']['cramers_v']:.4f}")
    
    return result

def test_recommendation_engine(df, schema):
    """Test recommendation engine"""
    print("\nTesting Recommendation Engine Module...")
    
    engine = RecommendationEngine()
    
    recommendations = engine.get_recommendations(df, schema)
    
    assert len(recommendations) > 0
    assert 'row_var' in recommendations[0]
    assert 'col_var' in recommendations[0]
    assert 'score' in recommendations[0]
    
    print("✓ Recommendation engine successful")
    print(f"  - Generated {len(recommendations)} recommendations")
    if recommendations:
        print(f"  - Top recommendation: {recommendations[0]['row_var']} × {recommendations[0]['col_var']}")
        print(f"  - Score: {recommendations[0]['score']:.1f}/100")
    
    return recommendations

def main():
    """Run all tests"""
    print("=" * 50)
    print("CONFIRM Data Converter - Module Tests")
    print("=" * 50)
    
    try:
        # Test data import
        df = test_data_import()
        
        # Test schema detection
        schema = test_schema_detection(df)
        
        # Test variable selection
        selector = test_variable_selection(df, schema)
        
        # Test contingency table generation
        result = test_contingency_generation(df)
        
        # Test recommendation engine
        recommendations = test_recommendation_engine(df, schema)
        
        print("\n" + "=" * 50)
        print("✅ All tests passed successfully!")
        print("=" * 50)
        
        # Display sample contingency table
        print("\nSample Contingency Table:")
        print(result['counts'])
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()