"""
Contingency Table Generator Module
Generates and validates contingency tables with statistical analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Tuple, List


class ContingencyTableGenerator:
    """Class for generating contingency tables with statistical validation"""
    
    def __init__(self):
        """Initialize the generator"""
        self.min_expected_freq = 5
        self.significance_level = 0.05
    
    def generate(self, 
                df: pd.DataFrame, 
                row_var: str, 
                col_var: str,
                normalize: Optional[str] = None,
                margins: bool = True,
                dropna: bool = True) -> Dict[str, Any]:
        """
        Generate contingency table with statistical analysis
        
        Args:
            df: DataFrame
            row_var: Row variable name
            col_var: Column variable name
            normalize: Normalization option ('all', 'index', 'columns', or None)
            margins: Whether to include margins
            dropna: Whether to drop NA values
            
        Returns:
            Dictionary containing contingency table and statistics
        """
        result = {
            'row_variable': row_var,
            'column_variable': col_var,
            'counts': None,
            'percentages': None,
            'row_percentages': None,
            'column_percentages': None,
            'statistics': {}
        }
        
        try:
            # Generate basic contingency table
            ct = pd.crosstab(
                df[row_var], 
                df[col_var], 
                margins=margins,
                margins_name='Total',
                dropna=dropna
            )
            
            result['counts'] = ct
            
            # Generate percentage tables
            if margins:
                # Remove margins for percentage calculations
                ct_no_margins = ct.iloc[:-1, :-1]
            else:
                ct_no_margins = ct
            
            # Overall percentages
            total = ct_no_margins.sum().sum()
            result['percentages'] = (ct_no_margins / total * 100).round(2)
            
            # Row percentages
            row_totals = ct_no_margins.sum(axis=1)
            result['row_percentages'] = ct_no_margins.div(row_totals, axis=0).multiply(100).round(2)
            
            # Column percentages
            col_totals = ct_no_margins.sum(axis=0)
            result['column_percentages'] = ct_no_margins.div(col_totals, axis=1).multiply(100).round(2)
            
            # Apply normalization if requested
            if normalize:
                if normalize == 'all':
                    normalized = ct_no_margins / ct_no_margins.sum().sum()
                elif normalize == 'index' or normalize == 'row':
                    normalized = ct_no_margins.div(ct_no_margins.sum(axis=1), axis=0)
                elif normalize == 'columns' or normalize == 'column':
                    normalized = ct_no_margins.div(ct_no_margins.sum(axis=0), axis=1)
                else:
                    normalized = ct_no_margins
                
                result['normalized'] = normalized
            
            # Calculate statistics
            stats_result = self._calculate_statistics(ct_no_margins)
            result['statistics'] = stats_result
            
            # Add interpretations
            result['interpretation'] = self._generate_interpretation(stats_result)
            
        except Exception as e:
            raise ValueError(f"Error generating contingency table: {str(e)}")
        
        return result
    
    def _calculate_statistics(self, ct: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate statistical measures for contingency table
        
        Args:
            ct: Contingency table (without margins)
            
        Returns:
            Dictionary with statistical measures
        """
        statistics = {}
        
        try:
            # Chi-square test
            chi2, p_value, dof, expected_freq = stats.chi2_contingency(ct)
            
            statistics['chi2'] = float(chi2)
            statistics['p_value'] = float(p_value)
            statistics['dof'] = int(dof)
            statistics['expected_freq'] = expected_freq
            
            # Check assumptions
            low_expected = (expected_freq < self.min_expected_freq).sum()
            total_cells = expected_freq.size
            statistics['low_expected_cells'] = int(low_expected)
            statistics['low_expected_percentage'] = (low_expected / total_cells) * 100
            
            # Cramér's V
            n = ct.sum().sum()
            min_dim = min(ct.shape[0] - 1, ct.shape[1] - 1)
            cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
            statistics['cramers_v'] = float(cramers_v)
            
            # Phi coefficient (for 2x2 tables)
            if ct.shape == (2, 2):
                phi = np.sqrt(chi2 / n)
                statistics['phi'] = float(phi)
            
            # Contingency coefficient
            contingency_coef = np.sqrt(chi2 / (chi2 + n))
            statistics['contingency_coefficient'] = float(contingency_coef)
            
            # Likelihood ratio
            g_test = 2 * np.sum(ct * np.log(ct / expected_freq + 1e-10))
            statistics['g_statistic'] = float(g_test)
            
            # Calculate residuals
            standardized_residuals = (ct - expected_freq) / np.sqrt(expected_freq)
            statistics['max_residual'] = float(np.abs(standardized_residuals).max())
            statistics['residuals'] = standardized_residuals.values.tolist()
            
            # Fisher's exact test for 2x2 tables
            if ct.shape == (2, 2):
                try:
                    odds_ratio, fisher_p = stats.fisher_exact(ct)
                    statistics['fisher_exact_p'] = float(fisher_p)
                    statistics['odds_ratio'] = float(odds_ratio)
                except:
                    pass
            
        except Exception as e:
            statistics['error'] = str(e)
        
        return statistics
    
    def _generate_interpretation(self, stats: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate interpretation of statistical results
        
        Args:
            stats: Statistics dictionary
            
        Returns:
            Dictionary with interpretations
        """
        interpretation = {}
        
        if 'p_value' in stats:
            p_value = stats['p_value']
            
            # Significance interpretation
            if p_value < 0.001:
                interpretation['significance'] = "Highly significant (p < 0.001)"
            elif p_value < 0.01:
                interpretation['significance'] = "Very significant (p < 0.01)"
            elif p_value < 0.05:
                interpretation['significance'] = "Significant (p < 0.05)"
            elif p_value < 0.1:
                interpretation['significance'] = "Marginally significant (p < 0.1)"
            else:
                interpretation['significance'] = "Not significant (p ≥ 0.1)"
            
            # Association interpretation
            if p_value < self.significance_level:
                interpretation['association'] = "There is a statistically significant association between the variables"
            else:
                interpretation['association'] = "There is no statistically significant association between the variables"
        
        if 'cramers_v' in stats:
            v = stats['cramers_v']
            
            # Effect size interpretation
            if v < 0.1:
                interpretation['effect_size'] = "Negligible association"
            elif v < 0.3:
                interpretation['effect_size'] = "Weak association"
            elif v < 0.5:
                interpretation['effect_size'] = "Moderate association"
            else:
                interpretation['effect_size'] = "Strong association"
        
        if 'low_expected_percentage' in stats:
            low_pct = stats['low_expected_percentage']
            
            # Assumption check
            if low_pct > 20:
                interpretation['assumption_warning'] = (
                    f"Warning: {low_pct:.1f}% of cells have expected frequency < {self.min_expected_freq}. "
                    "Chi-square test results may be unreliable. Consider Fisher's exact test or combining categories."
                )
            else:
                interpretation['assumption_check'] = "Chi-square test assumptions are satisfied"
        
        return interpretation
    
    def validate_table(self, ct: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate contingency table for statistical analysis
        
        Args:
            ct: Contingency table
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check for minimum size
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            validation['errors'].append("Table must have at least 2 rows and 2 columns")
            validation['is_valid'] = False
            return validation
        
        # Check for empty cells
        empty_cells = (ct == 0).sum().sum()
        if empty_cells > 0:
            empty_pct = (empty_cells / ct.size) * 100
            validation['warnings'].append(f"{empty_cells} empty cells ({empty_pct:.1f}%)")
            
            if empty_pct > 50:
                validation['recommendations'].append("Consider combining sparse categories")
        
        # Check for low frequencies
        low_freq_cells = (ct < 5).sum().sum()
        if low_freq_cells > 0:
            low_freq_pct = (low_freq_cells / ct.size) * 100
            validation['warnings'].append(f"{low_freq_cells} cells with count < 5 ({low_freq_pct:.1f}%)")
            
            if low_freq_pct > 20:
                validation['recommendations'].append(
                    "High proportion of low-frequency cells. Consider:\n"
                    "1. Combining categories\n"
                    "2. Using Fisher's exact test (for 2x2 tables)\n"
                    "3. Collecting more data"
                )
        
        # Check total sample size
        total = ct.sum().sum()
        if total < 20:
            validation['errors'].append(f"Insufficient sample size (n={total})")
            validation['is_valid'] = False
        elif total < 50:
            validation['warnings'].append(f"Small sample size (n={total})")
        
        # Check balance
        row_totals = ct.sum(axis=1)
        col_totals = ct.sum(axis=0)
        
        row_imbalance = row_totals.max() / row_totals.min() if row_totals.min() > 0 else float('inf')
        col_imbalance = col_totals.max() / col_totals.min() if col_totals.min() > 0 else float('inf')
        
        if row_imbalance > 10:
            validation['warnings'].append(f"Highly imbalanced row distribution (ratio: {row_imbalance:.1f})")
        
        if col_imbalance > 10:
            validation['warnings'].append(f"Highly imbalanced column distribution (ratio: {col_imbalance:.1f})")
        
        return validation
    
    def suggest_improvements(self, ct: pd.DataFrame) -> List[str]:
        """
        Suggest improvements for contingency table
        
        Args:
            ct: Contingency table
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Check for sparse categories that could be combined
        row_totals = ct.sum(axis=1)
        col_totals = ct.sum(axis=0)
        
        sparse_rows = row_totals[row_totals < 10]
        if len(sparse_rows) > 1:
            suggestions.append(
                f"Consider combining sparse row categories: {', '.join(map(str, sparse_rows.index))}"
            )
        
        sparse_cols = col_totals[col_totals < 10]
        if len(sparse_cols) > 1:
            suggestions.append(
                f"Consider combining sparse column categories: {', '.join(map(str, sparse_cols.index))}"
            )
        
        # Check for ordinal structure
        if ct.shape[0] > 2 and ct.shape[1] > 2:
            suggestions.append(
                "If variables are ordinal, consider using ordinal association measures "
                "(e.g., Kendall's tau, Spearman's rho)"
            )
        
        # Suggest visualization
        if ct.size <= 25:
            suggestions.append("Consider using a heatmap for visualization")
        elif ct.size <= 100:
            suggestions.append("Consider using a grouped bar chart for visualization")
        else:
            suggestions.append("Consider dimensionality reduction or focusing on key categories")
        
        return suggestions
    
    def export_for_confirm(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export contingency table in CONFIRM-compatible format
        
        Args:
            result: Contingency table result dictionary
            
        Returns:
            Dictionary formatted for CONFIRM validation
        """
        confirm_format = {
            'metadata': {
                'row_variable': result['row_variable'],
                'column_variable': result['column_variable'],
                'generated_timestamp': pd.Timestamp.now().isoformat(),
                'total_observations': int(result['counts'].iloc[:-1, :-1].sum().sum()) if 'Total' in result['counts'].index else int(result['counts'].sum().sum())
            },
            'contingency_table': result['counts'].to_dict(),
            'statistics': {
                'chi_square': result['statistics'].get('chi2'),
                'p_value': result['statistics'].get('p_value'),
                'degrees_of_freedom': result['statistics'].get('dof'),
                'cramers_v': result['statistics'].get('cramers_v'),
                'significance_level': self.significance_level
            },
            'validation': {
                'assumptions_met': result['statistics'].get('low_expected_percentage', 0) <= 20,
                'minimum_expected_frequency': self.min_expected_freq,
                'cells_below_minimum': result['statistics'].get('low_expected_cells', 0)
            }
        }
        
        return confirm_format