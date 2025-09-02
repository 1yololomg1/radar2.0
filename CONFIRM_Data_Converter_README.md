# CONFIRM Data Converter

Transform raw client data into properly formatted contingency tables suitable for statistical validation with CONFIRM.

## Overview

The CONFIRM Data Converter is a comprehensive tool that helps researchers and analysts transform raw data into properly formatted contingency tables for statistical analysis. It supports multiple input formats, provides intelligent variable selection guidance, and produces validated contingency tables with comprehensive statistical analysis.

## Features

### 1. Data Import Module
- **Multiple File Format Support**: 
  - Excel files (.xlsx, .xls)
  - CSV files (.csv)
  - JSON files (.json)
  - SQLite databases (.db, .sqlite, .sqlite3)
- **Data Preview**: Shows first 100 rows to help users understand data structure
- **Schema Detection**: Automatically identifies data types and potential categorical fields
- **Custom SQL Queries**: For database connections to pre-filter data
- **Encoding Detection**: Automatic detection for CSV files with various encodings

### 2. Variable Selection Interface
- **Interactive Column Selection**: User-friendly interface for selecting row and column variables
- **Data Type Analysis**: Automatic flagging of continuous variables that require categorization
- **Comprehensive Field Statistics**: 
  - Unique value counts
  - Min/max values
  - Null counts
  - Data type information
  - Cardinality ratios
- **Intelligent Recommendation Engine**: Suggests appropriate variables based on:
  - Cardinality analysis
  - Data distribution patterns
  - Statistical suitability for contingency analysis

### 3. Advanced Categorization Engine
- **Multiple Binning Methods**:
  - **Equal Width**: Divides range into equal-sized intervals
  - **Equal Frequency**: Creates bins with approximately equal number of observations
  - **Quantile**: Uses statistical quantiles (quartiles, quintiles, etc.)
  - **K-means**: Uses machine learning clustering for natural groupings
  - **Jenks Natural Breaks**: Optimizes bin boundaries to minimize within-group variance
  - **Custom**: User-defined bin edges
- **Smart Bin Suggestions**: Automatic recommendations using:
  - Sturges' rule
  - Square root rule
  - Rice rule
- **Interactive Bin Preview**: Real-time preview of binning results with distribution statistics

### 4. Contingency Table Generation & Validation
- **Robust Table Creation**: Handles missing data and edge cases
- **Comprehensive Statistical Validation**:
  - Chi-square test of independence
  - P-value calculation
  - Degrees of freedom
  - Expected frequency analysis
  - Assumption checking (minimum expected frequencies)
- **Quality Assurance**: Validates that tables meet statistical requirements

### 5. Visualization Suite
- **Multiple Chart Types**:
  - Heatmap visualization of contingency table
  - Bar charts showing marginal distributions
  - Stacked bar charts for proportional analysis
  - Percentage distribution heatmaps
- **Interactive Plots**: Integrated matplotlib visualizations
- **Export-Ready Graphics**: High-quality plots suitable for reports

### 6. Export & Reporting
- **Excel Export**: Complete contingency tables with validation statistics
- **CSV Export**: Simple table format for further analysis
- **Comprehensive Reports**: Detailed analysis reports (integrates with existing RADAR system)

## Installation

### Prerequisites
- Python 3.7 or higher
- Linux/Ubuntu system (tested on Ubuntu)

### Dependencies Installation
```bash
sudo apt update
sudo apt install -y python3-pandas python3-numpy python3-scipy python3-matplotlib python3-seaborn python3-openpyxl python3-sklearn python3-tk
```

### Running the Application
```bash
python3 launch_confirm_converter.py
```

Or directly:
```bash
python3 confirm_data_converter.py
```

## Usage Guide

### Step 1: Data Import
1. Launch the application
2. Go to the "Data Import" tab
3. Click "Browse" to select your data file
4. For database files, optionally enter a custom SQL query
5. Click "Import" to load your data
6. Review the data preview to ensure correct import

### Step 2: Variable Selection
1. Switch to the "Variable Selection" tab
2. Review the automatically generated variable statistics
3. Select your row variable (X-axis) from the dropdown
4. Select your column variable (Y-axis) from the dropdown
5. Review recommendations for variable suitability
6. Note any warnings about continuous variables requiring categorization

### Step 3: Contingency Table Generation
1. Go to the "Contingency Table" tab
2. Configure categorization settings:
   - Choose binning method for continuous variables
   - Set number of bins (2-20)
   - Use "Custom Bins" for advanced configuration
3. Click "Generate Table" to create the contingency table
4. Review the generated table, visualizations, and validation statistics

### Step 4: Export Results
1. Use "Export to Excel" for comprehensive results with statistics
2. Use "Export to CSV" for simple table format
3. Use "Generate Report" for detailed analysis reports

## Sample Data

The package includes sample datasets for testing:

### sample_data.csv
- Customer data with demographics and purchase information
- Mixed categorical and continuous variables
- 50 rows, 8 columns
- Good for testing basic functionality

### sample_survey_data.json
- Survey response data with satisfaction ratings
- Complex nested structure
- 15 rows, 9 columns
- Good for testing JSON import and advanced categorization

## Data Requirements

### Input Data Format
- **Structured Data**: Tabular format with rows as observations and columns as variables
- **Variable Types**: Supports both categorical and continuous variables
- **Missing Data**: Handles null/missing values appropriately
- **Size**: Optimized for datasets from small (dozens of rows) to medium (thousands of rows)

### Recommended Data Characteristics
- **Minimum Sample Size**: At least 30 observations for reliable statistical analysis
- **Variable Cardinality**: 
  - Categorical variables: 2-50 unique categories
  - Continuous variables: Will be automatically binned
- **Data Quality**: Less than 50% missing values per variable for best results

## Statistical Validation

The tool automatically performs comprehensive statistical validation:

### Chi-Square Test of Independence
- Tests the null hypothesis that variables are independent
- Provides chi-square statistic, p-value, and degrees of freedom
- Interprets results with clear significance indicators

### Assumption Checking
- **Minimum Expected Frequency**: Ensures all expected frequencies ≥ 1
- **5-Count Rule**: Checks that no more than 20% of cells have expected frequency < 5
- **Sample Size**: Validates adequate sample size for reliable results

### Quality Indicators
- Color-coded validation results
- Clear warnings for assumption violations
- Recommendations for improving analysis quality

## Advanced Features

### Custom Binning Dialog
- Interactive interface for configuring continuous variable categorization
- Real-time preview of binning results
- Statistical guidance for optimal bin selection
- Support for domain-specific binning requirements

### Intelligent Recommendations
- Automatic variable type detection
- Suitability scoring for contingency analysis
- Warnings for potential data quality issues
- Suggestions for improving analysis

### Robust Error Handling
- Graceful handling of malformed data
- Clear error messages with actionable guidance
- Automatic fallback methods for edge cases

## Integration

The CONFIRM Data Converter is designed to integrate with existing analysis workflows:

- **RADAR Integration**: Compatible with the existing RADAR analysis system
- **Export Formats**: Multiple export options for downstream analysis
- **API Design**: Modular architecture allows for programmatic usage

## Troubleshooting

### Common Issues

1. **Import Errors**: 
   - Check file format and encoding
   - Ensure file is not corrupted or locked
   - Verify file permissions

2. **Categorization Issues**:
   - Ensure continuous variables have sufficient range
   - Check for extreme outliers that might affect binning
   - Use custom bins for domain-specific requirements

3. **Statistical Warnings**:
   - Low expected frequencies: Consider combining categories or collecting more data
   - Independence assumption: Review variable selection and data collection methodology

### Performance Notes
- Large datasets (>10,000 rows) may take longer to process
- Complex visualizations may require additional memory
- Database queries should be optimized for performance

## Technical Architecture

### Core Components
1. **DataImportModule**: Handles multi-format data import with robust error handling
2. **VariableSelectionInterface**: Provides intelligent variable selection with recommendations
3. **CategorizationEngine**: Advanced binning algorithms for continuous variables
4. **ContingencyTableGenerator**: Creates and validates contingency tables
5. **CustomBinsDialog**: Interactive interface for advanced categorization

### Design Principles
- **Modularity**: Each component can be used independently
- **Extensibility**: Easy to add new import formats or categorization methods
- **Robustness**: Comprehensive error handling and validation
- **User-Friendly**: Intuitive interface with intelligent guidance

## Version History

### Version 1.0.0
- Initial release with core functionality
- Support for CSV, Excel, JSON, and SQLite import
- Multiple categorization methods
- Comprehensive statistical validation
- Interactive visualization suite
- Export capabilities

## License

Commercial software - Professional Analysis Software Suite
Copyright (c) 2025 CONFIRM Development Team. All rights reserved.

## Support

For technical support or feature requests, contact the CONFIRM Development Team.