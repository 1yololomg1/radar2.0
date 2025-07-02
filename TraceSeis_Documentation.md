# TraceSeis SeisTool - Geophysical Model Validator

## Professional Neural Network Classification Analysis Tool

**Version:** 2.0.0  
**License:** Commercial - TraceSeis SeisTool Suite  
**Copyright:** © 2025 TraceSeis. All rights reserved.

---

## Overview

The TraceSeis Geophysical Model Validator is an industry-grade application designed specifically for the oil & gas industry to analyze and validate neural network classification results for lithofacies prediction. This tool provides comprehensive statistical analysis and professional visualizations for geophysical modeling workflows.

## Key Features

### 🔬 **Advanced Statistical Analysis**
- **Confusion Matrix Analysis**: Automated conversion from contingency tables
- **Global Fit Calculation**: Weighted accuracy considering class distribution
- **Cramér's V**: Association strength measurement between predicted and actual classifications
- **Percent Undefined**: Quantification of ambiguous classifications
- **Per-Class Metrics**: Precision, Recall, F1-Score for each lithofacies

### 📊 **Professional Visualizations**
- **Confusion Matrix Heatmaps**: Clear visualization of classification performance
- **Distribution Pie Charts**: Lithofacies distribution analysis
- **Radar Charts**: Multi-model performance comparison
- **Interactive Navigation**: Zoom, pan, and export capabilities

### 🏭 **Industry-Grade Features**
- **Multi-Model Analysis**: Compare different neural network configurations
- **Robust Error Handling**: Comprehensive validation and error reporting
- **Professional Logging**: Detailed operation logs for troubleshooting
- **Export Capabilities**: Results in multiple formats (TXT, CSV, PNG)
- **Thread-Safe Operations**: Non-blocking analysis with progress tracking

---

## Installation

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows 10/11, macOS 10.14+, Linux
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Disk Space**: 500MB for installation + data storage

### Installation Steps

1. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Installation**
   ```bash
   python traceseis_validator.py --version
   ```

3. **Create Desktop Shortcut** (Optional)
   ```bash
   python create_shortcut.py
   ```

---

## Usage Guide

### Data Format Requirements

The application expects Excel files with the following structure:

```
Sheet Name: Neural Network Configuration (e.g., "6x6", "8x8", "10x10")

| Neuron | FineSand | MedFineSnd | MedCoarseSnd | SandAndShale |
|--------|----------|------------|--------------|--------------|
| 1      | 78       | 4          | 0            | 18           |
| 2      | 11       | 50         | 39           | 0            |
| 3      | 0        | 0          | 0            | 0            |
| ...    | ...      | ...        | ...          | ...          |
```

**Key Requirements:**
- First column: Neuron identifier
- Subsequent columns: Percentage classification for each lithofacies
- Row percentages should sum to approximately 100%
- Each sheet represents a different neural network configuration

### Workflow

1. **Load Data**
   - Click "Browse" to select your Excel file
   - Available models (sheets) will appear in the list

2. **Select Models**
   - Choose one or more neural network configurations
   - Use Ctrl+Click for multiple selections

3. **Run Analysis**
   - Click "Analyze Selected Models"
   - Progress will be shown in the status bar

4. **Review Results**
   - **Analysis Panel**: Detailed statistical metrics
   - **Visualization Panel**: Interactive charts and plots
   - **Comparison**: Radar chart for multi-model analysis

5. **Export Results**
   - Click "Export Results" to save analysis
   - Choose format: Text, CSV, or combined package

---

## Statistical Methodology

### Global Fit Calculation
```
Global Fit = Σ(class_weight_i × class_accuracy_i) × 100%

Where:
- class_weight_i = proportion of samples in class i
- class_accuracy_i = diagonal_element_i / row_sum_i
```

### Cramér's V Calculation
```
Cramér's V = √(χ² / (n × min(r-1, c-1)))

Where:
- χ² = Chi-square statistic
- n = total sample size
- r, c = number of rows and columns
```

### Percent Undefined
```
Undefined% = (neurons with max_probability < threshold) / total_neurons × 100%

Default threshold = 10%
```

---

## Configuration

### Application Settings

Create `config.yaml` in the application directory:

```yaml
# TraceSeis Model Validator Configuration
application:
  name: "TraceSeis SeisTool - Model Validator"
  version: "2.0.0"
  log_level: "INFO"

analysis:
  undefined_threshold: 0.10  # 10% threshold for undefined classifications
  max_file_size_mb: 100      # Maximum Excel file size
  validation_tolerance: 5.0   # Tolerance for row sum validation (%)

visualization:
  default_dpi: 300           # Export resolution
  figure_size: [12, 8]       # Default figure size
  color_palette: "husl"      # Seaborn color palette

export:
  include_figures: true      # Export figures with results
  figure_format: "png"       # Default figure format
  csv_precision: 3           # Decimal places in CSV export
```

### Logging Configuration

```yaml
logging:
  version: 1
  disable_existing_loggers: false
  formatters:
    standard:
      format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  handlers:
    file:
      class: logging.FileHandler
      filename: traceseis_validator.log
      formatter: standard
      level: INFO
    console:
      class: logging.StreamHandler
      formatter: standard
      level: WARNING
  root:
    level: INFO
    handlers: [file, console]
```

---

## API Reference

### Core Classes

#### `StatisticalAnalyzer`
Primary analysis engine for neural network validation.

**Methods:**
- `validate_contingency_data(data)`: Validate input data format
- `convert_to_confusion_matrix(data)`: Convert percentages to confusion matrix
- `calculate_global_fit(matrix)`: Calculate global fit metric
- `calculate_cramers_v(matrix)`: Calculate Cramér's V statistic
- `analyze_model(data, name)`: Complete model analysis

#### `VisualizationEngine`
Professional visualization creation and management.

**Methods:**
- `create_confusion_matrix_plot(metrics)`: Generate confusion matrix heatmap
- `create_pie_chart(metrics)`: Generate distribution pie chart
- `create_radar_chart(metrics_list)`: Generate comparison radar chart

#### `ModelMetrics`
Data container for analysis results.

**Attributes:**
- `model_name`: Neural network configuration name
- `global_fit`: Global fit percentage
- `cramers_v`: Cramér's V statistic
- `percent_undefined`: Percentage of undefined classifications
- `accuracy`: Overall classification accuracy
- `precision`: Per-class precision scores
- `recall`: Per-class recall scores

---

## Troubleshooting

### Common Issues

**Issue**: "Invalid data format" error
**Solution**: Ensure Excel data has numeric values in classification columns and row sums ≈ 100%

**Issue**: Visualization not displaying
**Solution**: Check matplotlib backend and ensure GUI libraries are properly installed

**Issue**: Memory errors with large files
**Solution**: Increase `max_file_size_mb` in config or process files in smaller batches

**Issue**: Export failures
**Solution**: Verify write permissions in target directory and sufficient disk space

### Debug Mode

Enable debug logging by setting environment variable:
```bash
export TRACESEIS_DEBUG=1
python traceseis_validator.py
```

### Log File Analysis

Check `traceseis_validator.log` for detailed operation logs:
- Application startup and shutdown
- Data loading and validation
- Analysis progress and results
- Error details and stack traces

---

## Support & Contact

**TraceSeis Development Team**  
Email: support@traceseis.com  
Documentation: https://docs.traceseis.com/seitool  
Support Portal: https://support.traceseis.com

**License Information**  
This software is licensed for commercial use under the TraceSeis SeisTool Suite license.  
License Key Required: Contact sales@traceseis.com

---

## Changelog

### Version 2.0.0 (Current)
- Complete rewrite for professional deployment
- Industry-grade statistical analysis engine
- Professional GUI with advanced visualizations
- Multi-model comparison capabilities
- Comprehensive export functionality
- Robust error handling and logging

### Version 1.0.0 (Legacy)
- Initial prototype implementation
- Basic contingency table analysis
- Simple visualization capabilities

---

## Technical Specifications

### Performance Benchmarks
- **File Loading**: 10MB Excel file in <5 seconds
- **Analysis**: 4 models (100 neurons each) in <10 seconds
- **Visualization**: Chart generation in <3 seconds
- **Memory Usage**: <500MB for typical workflows

### Compatibility Matrix

| Component | Version | Status |
|-----------|---------|--------|
| Python | 3.8+ | ✅ Supported |
| NumPy | 1.21+ | ✅ Required |
| Pandas | 1.3+ | ✅ Required |
| Matplotlib | 3.5+ | ✅ Required |
| Windows | 10/11 | ✅ Tested |
| macOS | 10.14+ | ✅ Tested |
| Linux | Ubuntu 20.04+ | ✅ Tested |

---

*TraceSeis SeisTool - Advancing Geophysical Analysis through Professional Software Solutions*
