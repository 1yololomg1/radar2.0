#!/usr/bin/env python3
"""
Test GUI components without launching full interface
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all GUI components can be imported"""
    try:
        from confirm_data_converter import (
            DataImportModule, 
            VariableSelectionInterface, 
            CategorizationEngine,
            ContingencyTableGenerator,
            CONFIRMDataConverter,
            CustomBinsDialog
        )
        print("✓ All components imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_data_processing():
    """Test core data processing without GUI"""
    try:
        from confirm_data_converter import DataImportModule, ContingencyTableConfig, ContingencyTableGenerator
        
        # Test data import
        importer = DataImportModule()
        df = importer.import_data('sample_data.csv')
        
        # Test contingency table generation
        generator = ContingencyTableGenerator()
        config = ContingencyTableConfig(
            row_variable='education_level',
            column_variable='purchase_category'
        )
        
        table, stats = generator.generate_table(df, config)
        
        print("✓ Core data processing works correctly")
        print(f"  Generated table shape: {table.shape}")
        print(f"  Chi-square statistic: {stats.get('chi2_stat', 'N/A'):.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data processing error: {e}")
        return False

def main():
    """Run component tests"""
    print("CONFIRM Data Converter - Component Tests")
    print("=" * 50)
    
    success = True
    
    # Test imports
    success &= test_imports()
    
    # Test data processing
    success &= test_data_processing()
    
    if success:
        print("\n✓ All component tests passed!")
        print("The CONFIRM Data Converter is ready for use.")
    else:
        print("\n✗ Some tests failed. Please check the error messages above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)