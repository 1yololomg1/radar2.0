# CONFIRM Data Converter

A comprehensive Python application for transforming raw client data into properly formatted contingency tables suitable for statistical validation with CONFIRM.

## Features

### 1. **Data Import Module**
- **Multiple Format Support**: Excel (.xlsx, .xls), CSV, JSON, SQL databases
- **Database Connections**: PostgreSQL, MySQL, SQLite
- **Smart Encoding Detection**: Automatically handles different file encodings
- **Data Preview**: Shows first 100 rows with search functionality
- **Custom SQL Queries**: Pre-filter data directly from databases

### 2. **Schema Detection**
- **Automatic Type Detection**: Identifies categorical, numeric, datetime, and text fields
- **Intelligent Categorization**: Determines if numeric fields should be treated as categorical
- **Data Quality Metrics**: Completeness scores, uniqueness analysis, missing value detection
- **Sample Value Display**: Shows representative values for each column

### 3. **Variable Selection Interface**
- **Smart Recommendations**: AI-powered suggestions for variable pairs based on:
  - Statistical association strength
  - Semantic relationships
  - Domain relevance
  - Data completeness
- **Field Statistics**: Detailed statistics for each variable including:
  - Unique value counts
  - Distribution visualizations
  - Missing value analysis
  - Mode and frequency information
- **Continuous Variable Categorization**: Multiple methods for binning continuous data:
  - Equal width binning
  - Equal frequency (quantile) binning
  - Custom breakpoints
  - Domain-specific suggestions

### 4. **Contingency Table Generation**
- **Multiple Views**: 
  - Raw counts
  - Percentages (overall, row-wise, column-wise)
  - Normalized tables
- **Statistical Analysis**:
  - Chi-square test
  - Cramér's V
  - Fisher's exact test (for 2x2 tables)
  - Contingency coefficient
  - Standardized residuals
- **Visualizations**:
  - Interactive heatmaps
  - Grouped/stacked bar charts
  - 3D surface plots
- **Validation**: Automatic checking of statistical assumptions

### 5. **Export Options**
- **Multiple Formats**: CSV, Excel, JSON, LaTeX, Markdown
- **CONFIRM Compatible**: Specially formatted output for CONFIRM validation
- **Include Statistics**: Optional inclusion of statistical results in exports

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd confirm-converter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
streamlit run confirm_converter.py
```

The application will open in your default web browser at `http://localhost:8501`

### Step-by-Step Guide

1. **Import Your Data**
   - Choose import method: File upload, database connection, or sample data
   - For files: Select CSV, Excel, or JSON
   - For databases: Enter connection details and SQL query
   - Review the automatic schema detection

2. **Review Data Quality**
   - Check the Data Preview tab to see your data
   - Review the Schema tab for data type detection
   - Check Statistics for detailed column analysis
   - Review Data Quality scores and warnings

3. **Select Variables**
   - Review AI-powered recommendations
   - Or manually select row and column variables
   - Categorize continuous variables if needed
   - Check variable statistics and distributions

4. **Generate Contingency Table**
   - Choose normalization options
   - Generate the table
   - Review multiple visualization options
   - Check statistical analysis results

5. **Export Results**
   - Choose export format
   - Include/exclude statistical results
   - Download formatted output

## Sample Data

The application includes three sample datasets for testing:

1. **Sales Data**: Customer transactions with regions, products, and satisfaction scores
2. **Survey Results**: Demographic data with preferences and responses
3. **Medical Records**: Patient data with diagnoses, treatments, and outcomes

## Technical Architecture

```
confirm_converter.py       # Main Streamlit application
├── data_import.py        # Handles file imports and database connections
├── schema_detector.py    # Automatic data type detection
├── variable_selector.py  # Variable analysis and selection logic
├── contingency_generator.py # Table generation and statistics
└── recommendation_engine.py # AI-powered recommendations
```

## Statistical Methods

### Association Measures
- **Chi-square test**: Tests independence between variables
- **Cramér's V**: Effect size for nominal associations (0-1 scale)
- **Phi coefficient**: Association measure for 2x2 tables
- **Contingency coefficient**: Alternative association measure

### Interpretation Guidelines
- **Cramér's V**:
  - < 0.1: Negligible association
  - 0.1-0.3: Weak association
  - 0.3-0.5: Moderate association
  - \> 0.5: Strong association

- **P-values**:
  - < 0.001: Highly significant
  - < 0.01: Very significant
  - < 0.05: Significant
  - ≥ 0.05: Not significant

## Requirements

- Python 3.8+
- See `requirements.txt` for full package list

## Key Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scipy**: Statistical tests
- **plotly**: Interactive visualizations
- **openpyxl/xlrd**: Excel file handling
- **sqlalchemy**: Database connections

## Tips for Best Results

1. **Data Preparation**:
   - Ensure categorical variables have meaningful labels
   - Handle missing values appropriately
   - Check for data entry errors

2. **Variable Selection**:
   - Choose variables with 3-10 categories for best results
   - Ensure adequate sample size (>50 observations)
   - Consider semantic relationships between variables

3. **Statistical Validity**:
   - Check that >80% of expected frequencies are ≥5
   - For small samples, use Fisher's exact test
   - Consider combining sparse categories

## Troubleshooting

### Common Issues

1. **Import Errors**:
   - Check file encoding (UTF-8 recommended)
   - Ensure column names don't have special characters
   - Verify database connection credentials

2. **Statistical Warnings**:
   - "Low expected frequencies": Consider combining categories
   - "Sparse table": Need more data or fewer categories
   - "No association found": Variables may be independent

3. **Performance**:
   - Large datasets (>100k rows) may be slow
   - Consider sampling or filtering data first
   - Use database queries to pre-aggregate if needed

## License

MIT License - See LICENSE file for details

## Support

For issues, questions, or feature requests, please open an issue on the GitHub repository.