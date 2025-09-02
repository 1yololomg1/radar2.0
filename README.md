# CONFIRM Data Converter

A comprehensive tool to transform raw client data into properly formatted contingency tables suitable for statistical validation with CONFIRM. The application supports multiple input formats, provides intelligent data analysis, and generates validated contingency tables with detailed reporting.

## Features

### Data Import Module
- **Multi-format Support**: Excel (.xlsx), CSV (.csv), JSON (.json)
- **Database Connections**: PostgreSQL, MySQL, SQLite, SQL Server
- **Data Preview**: Interactive preview of first 100 rows
- **Schema Detection**: Automatic data type identification and categorical field detection
- **Custom Queries**: Pre-filter data with custom SQL queries for database connections

### Variable Selection Interface
- **Smart Column Selection**: Intuitive row and column variable selection
- **Data Type Analysis**: Automatic detection of continuous vs categorical variables
- **Field Statistics**: Comprehensive statistics including unique counts, null values, min/max
- **Recommendation Engine**: AI-powered suggestions based on cardinality and distribution
- **Validation Warnings**: Alerts for continuous variables that may need categorization

### Contingency Table Generation
- **Automatic Table Creation**: Generate contingency tables from selected variables
- **CONFIRM Validation**: Built-in validation rules for statistical analysis requirements
- **Quality Checks**: Minimum cell count validation, empty row/column detection
- **Export Options**: Download as CSV or comprehensive Excel reports

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd confirm-data-converter
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run confirm_data_converter.py
   ```

## Usage

### 1. Data Import
- **File Upload**: Select your data file (Excel, CSV, or JSON) using the sidebar
- **Database Connection**: Connect to your database using connection strings
- **Data Preview**: Review the imported data structure and statistics

### 2. Variable Analysis
- **Schema Review**: Examine detected data types and categorical suggestions
- **Field Statistics**: View detailed statistics for each variable
- **Recommendations**: Follow AI suggestions for optimal variable selection

### 3. Contingency Table Creation
- **Variable Selection**: Choose row and column variables for your analysis
- **Table Generation**: Create the contingency table automatically
- **Validation**: Review validation results and recommendations
- **Export**: Download results in CSV or Excel format

## Supported Data Formats

### File Formats
- **Excel (.xlsx)**: Full support with multi-sheet selection
- **CSV (.csv)**: Automatic encoding detection (UTF-8, Latin-1, CP1252, ISO-8859-1)
- **JSON (.json)**: Flexible structure handling (arrays, objects, nested data)

### Database Connections
- **PostgreSQL**: `postgresql://user:password@host:port/database`
- **MySQL**: `mysql+pymysql://user:password@host:port/database`
- **SQLite**: `sqlite:///path/to/database.db`
- **SQL Server**: `mssql+pyodbc://user:password@host:port/database?driver=ODBC+Driver+17+for+SQL+Server`

## Validation Rules

The application validates contingency tables according to CONFIRM requirements:

- **Minimum Cell Count**: Warns about cells with counts < 5
- **Table Dimensions**: Ensures at least 2 rows and 2 columns
- **Empty Rows/Columns**: Detects and reports empty categories
- **Data Quality**: Identifies potential issues with variable selection

## Example Workflow

1. **Import Data**: Upload a CSV file with survey responses
2. **Review Schema**: System detects "Age_Group" and "Satisfaction_Level" as categorical
3. **Select Variables**: Choose "Age_Group" as row variable and "Satisfaction_Level" as column variable
4. **Generate Table**: Create contingency table showing age group vs satisfaction distribution
5. **Validate**: Review warnings about low cell counts in some categories
6. **Export**: Download Excel report with table and validation results

## Technical Details

### Architecture
- **Frontend**: Streamlit for interactive web interface
- **Data Processing**: Pandas for data manipulation and analysis
- **Database**: SQLAlchemy for database connectivity
- **Visualization**: Plotly for interactive charts and tables
- **Validation**: Custom validation engine for CONFIRM compliance

### Performance
- **Memory Efficient**: Handles large datasets with optimized pandas operations
- **Fast Processing**: Efficient algorithms for schema detection and table generation
- **Scalable**: Supports datasets with millions of rows

## Troubleshooting

### Common Issues

1. **File Encoding Errors**: The application automatically tries multiple encodings for CSV files
2. **Database Connection**: Ensure connection strings are properly formatted
3. **Memory Issues**: For very large files, consider pre-filtering data
4. **Missing Dependencies**: Run `pip install -r requirements.txt` to install all required packages

### Error Messages
- **"No data to preview"**: Check file format and content
- **"Database connection error"**: Verify connection string and credentials
- **"Table validation errors"**: Review variable selection and data quality

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the example data files for reference

## Changelog

### Version 1.0.0
- Initial release with full CONFIRM Data Converter functionality
- Support for Excel, CSV, JSON, and database imports
- Intelligent schema detection and variable recommendations
- Comprehensive contingency table validation
- Export capabilities for CSV and Excel formats