"""
Demo script for CONFIRM Data Converter
Shows how to use the converter programmatically
"""

import pandas as pd
import numpy as np
from data_import import DataImporter
from schema_detector import SchemaDetector
from variable_selector import VariableSelector
from contingency_generator import ContingencyTableGenerator
from recommendation_engine import RecommendationEngine

def create_demo_data():
    """Create a sample dataset for demonstration"""
    np.random.seed(42)
    n = 500
    
    data = {
        'Age_Group': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], n, p=[0.15, 0.25, 0.25, 0.20, 0.15]),
        'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n, p=[0.3, 0.4, 0.25, 0.05]),
        'Employment': np.random.choice(['Employed', 'Self-employed', 'Unemployed', 'Student'], n, p=[0.5, 0.2, 0.1, 0.2]),
        'Income_Level': np.random.choice(['<30k', '30-60k', '60-100k', '>100k'], n, p=[0.25, 0.35, 0.25, 0.15]),
        'Product_Preference': np.random.choice(['Product A', 'Product B', 'Product C'], n, p=[0.4, 0.35, 0.25]),
        'Satisfaction': np.random.choice(['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied'], n, p=[0.3, 0.4, 0.2, 0.1]),
        'Purchase_Frequency': np.random.choice(['Weekly', 'Monthly', 'Quarterly', 'Yearly'], n, p=[0.1, 0.4, 0.3, 0.2])
    }
    
    return pd.DataFrame(data)

def main():
    print("=" * 70)
    print("CONFIRM DATA CONVERTER - DEMONSTRATION")
    print("=" * 70)
    
    # Step 1: Create demo data
    print("\n📊 Step 1: Creating demo dataset...")
    df = create_demo_data()
    print(f"✓ Created dataset with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {', '.join(df.columns)}")
    
    # Step 2: Detect schema
    print("\n🔍 Step 2: Detecting data schema...")
    detector = SchemaDetector()
    schema = detector.detect_schema(df)
    
    print("✓ Schema detected:")
    for col, info in schema.items():
        print(f"  • {col}: {info['data_type']} ({info['unique_values']} unique values)")
    
    # Step 3: Get recommendations
    print("\n💡 Step 3: Getting variable pair recommendations...")
    engine = RecommendationEngine()
    recommendations = engine.get_recommendations(df, schema)
    
    print(f"✓ Top 3 recommended variable pairs:")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"  {i}. {rec['row_var']} × {rec['col_var']} (Score: {rec['score']:.1f}/100)")
        if rec['reasons']:
            print(f"     Reasons: {', '.join(rec['reasons'][:2])}")
    
    # Step 4: Analyze variable pair
    print("\n📈 Step 4: Analyzing selected variable pair...")
    row_var = 'Education'
    col_var = 'Income_Level'
    
    selector = VariableSelector()
    pair_analysis = selector.analyze_pair(df, row_var, col_var)
    
    print(f"✓ Analysis for {row_var} × {col_var}:")
    print(f"  • Table shape: {pair_analysis['table_shape']}")
    print(f"  • Complete cases: {pair_analysis['complete_cases']}")
    print(f"  • Completeness: {pair_analysis['completeness']:.1f}%")
    
    # Step 5: Generate contingency table
    print("\n📊 Step 5: Generating contingency table...")
    generator = ContingencyTableGenerator()
    result = generator.generate(df, row_var, col_var, margins=True)
    
    print(f"✓ Contingency table generated:")
    print("\nCounts:")
    print(result['counts'])
    
    # Step 6: Display statistics
    print("\n📉 Step 6: Statistical Analysis:")
    stats = result['statistics']
    print(f"  • Chi-square statistic: {stats['chi2']:.4f}")
    print(f"  • P-value: {stats['p_value']:.6f}")
    print(f"  • Degrees of freedom: {stats['dof']}")
    print(f"  • Cramér's V: {stats['cramers_v']:.4f}")
    
    # Interpretation
    interpretation = result['interpretation']
    print(f"\n📝 Interpretation:")
    print(f"  • {interpretation['significance']}")
    print(f"  • {interpretation['effect_size']}")
    print(f"  • {interpretation['association']}")
    
    # Step 7: Export options
    print("\n💾 Step 7: Export Options:")
    print("  • CSV format: contingency_table.csv")
    print("  • Excel format: contingency_table.xlsx")
    print("  • JSON format: contingency_table.json")
    print("  • LaTeX format: contingency_table.tex")
    print("  • Markdown format: contingency_table.md")
    
    # Save example CSV
    result['counts'].to_csv('/tmp/demo_contingency_table.csv')
    print("\n✓ Sample output saved to /tmp/demo_contingency_table.csv")
    
    print("\n" + "=" * 70)
    print("🎉 Demo completed successfully!")
    print("=" * 70)
    print("\n📌 To use the full application with GUI, run:")
    print("   streamlit run confirm_converter.py")
    print("\n📌 The application provides:")
    print("   • Interactive data import from multiple sources")
    print("   • Automatic schema detection and validation")
    print("   • Smart variable recommendations")
    print("   • Interactive visualizations")
    print("   • Comprehensive statistical analysis")
    print("   • Multiple export formats")

if __name__ == "__main__":
    main()