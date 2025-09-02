"""
Variable Selection Module
Handles variable selection and field statistics
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class VariableSelector:
    """Class for variable selection and analysis"""
    
    def __init__(self):
        """Initialize the variable selector"""
        self.min_categories = 2
        self.max_categories = 50
        self.min_observations = 5
    
    def analyze_variable(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Analyze a single variable for contingency table suitability
        
        Args:
            df: DataFrame
            column: Column name to analyze
            
        Returns:
            Dictionary with variable analysis
        """
        series = df[column]
        
        analysis = {
            'column': column,
            'total_count': len(series),
            'non_null_count': series.notna().sum(),
            'null_count': series.isna().sum(),
            'null_percentage': (series.isna().sum() / len(series)) * 100,
            'unique_count': series.nunique(),
            'unique_percentage': (series.nunique() / len(series)) * 100,
            'is_suitable': False,
            'warnings': [],
            'value_distribution': {}
        }
        
        # Get value counts
        value_counts = series.value_counts()
        
        # Check suitability for contingency table
        if analysis['unique_count'] < self.min_categories:
            analysis['warnings'].append(f"Too few categories ({analysis['unique_count']} < {self.min_categories})")
        elif analysis['unique_count'] > self.max_categories:
            analysis['warnings'].append(f"Too many categories ({analysis['unique_count']} > {self.max_categories})")
        else:
            analysis['is_suitable'] = True
        
        # Check for sparse categories
        if len(value_counts) > 0:
            min_count = value_counts.min()
            if min_count < self.min_observations:
                sparse_categories = value_counts[value_counts < self.min_observations].index.tolist()
                analysis['warnings'].append(f"Sparse categories with < {self.min_observations} observations: {len(sparse_categories)}")
        
        # Store value distribution
        if analysis['unique_count'] <= 20:
            analysis['value_distribution'] = value_counts.to_dict()
        else:
            # Store top 10 and bottom 5
            analysis['value_distribution'] = {
                'top_10': value_counts.head(10).to_dict(),
                'bottom_5': value_counts.tail(5).to_dict(),
                'others_count': len(value_counts) - 15
            }
        
        # Calculate statistics based on data type
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                analysis['numeric_stats'] = {
                    'mean': float(non_null.mean()),
                    'median': float(non_null.median()),
                    'std': float(non_null.std()),
                    'min': float(non_null.min()),
                    'max': float(non_null.max()),
                    'q1': float(non_null.quantile(0.25)),
                    'q3': float(non_null.quantile(0.75))
                }
        
        # Calculate mode
        if len(value_counts) > 0:
            analysis['mode'] = value_counts.index[0]
            analysis['mode_count'] = int(value_counts.iloc[0])
            analysis['mode_percentage'] = (value_counts.iloc[0] / analysis['non_null_count']) * 100
        
        return analysis
    
    def analyze_pair(self, df: pd.DataFrame, row_var: str, col_var: str) -> Dict[str, Any]:
        """
        Analyze a pair of variables for contingency table
        
        Args:
            df: DataFrame
            row_var: Row variable name
            col_var: Column variable name
            
        Returns:
            Dictionary with pair analysis
        """
        analysis = {
            'row_variable': row_var,
            'column_variable': col_var,
            'is_valid': True,
            'warnings': [],
            'table_shape': None,
            'expected_cells': 0,
            'sparse_cells': 0,
            'completeness': 0
        }
        
        # Check if variables exist
        if row_var not in df.columns:
            analysis['is_valid'] = False
            analysis['warnings'].append(f"Row variable '{row_var}' not found in data")
            return analysis
        
        if col_var not in df.columns:
            analysis['is_valid'] = False
            analysis['warnings'].append(f"Column variable '{col_var}' not found in data")
            return analysis
        
        # Check if variables are different
        if row_var == col_var:
            analysis['is_valid'] = False
            analysis['warnings'].append("Row and column variables must be different")
            return analysis
        
        # Get complete cases
        complete_mask = df[[row_var, col_var]].notna().all(axis=1)
        complete_cases = complete_mask.sum()
        total_cases = len(df)
        
        analysis['completeness'] = (complete_cases / total_cases) * 100
        analysis['complete_cases'] = complete_cases
        analysis['missing_cases'] = total_cases - complete_cases
        
        if complete_cases < 10:
            analysis['is_valid'] = False
            analysis['warnings'].append(f"Too few complete cases ({complete_cases})")
            return analysis
        
        # Create contingency table
        try:
            ct = pd.crosstab(df[row_var], df[col_var])
            analysis['table_shape'] = ct.shape
            analysis['expected_cells'] = ct.size
            
            # Check for sparse cells
            sparse_mask = ct < self.min_observations
            analysis['sparse_cells'] = sparse_mask.sum().sum()
            analysis['sparse_percentage'] = (analysis['sparse_cells'] / analysis['expected_cells']) * 100
            
            if analysis['sparse_percentage'] > 20:
                analysis['warnings'].append(f"High sparsity: {analysis['sparse_percentage']:.1f}% of cells have < {self.min_observations} observations")
            
            # Calculate expected frequencies under independence
            row_totals = ct.sum(axis=1)
            col_totals = ct.sum(axis=0)
            total = ct.sum().sum()
            
            expected = np.outer(row_totals, col_totals) / total
            
            # Check if expected frequencies are adequate
            low_expected = (expected < 5).sum()
            if low_expected > 0:
                low_expected_pct = (low_expected / expected.size) * 100
                if low_expected_pct > 20:
                    analysis['warnings'].append(f"{low_expected_pct:.1f}% of cells have expected frequency < 5 (chi-square test may be unreliable)")
            
            # Calculate association strength (Cramér's V preview)
            chi2 = ((ct - expected) ** 2 / expected).sum().sum()
            n = ct.sum().sum()
            min_dim = min(ct.shape[0] - 1, ct.shape[1] - 1)
            cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
            
            analysis['association_preview'] = {
                'cramers_v': float(cramers_v),
                'strength': self._interpret_cramers_v(cramers_v)
            }
            
        except Exception as e:
            analysis['is_valid'] = False
            analysis['warnings'].append(f"Error creating contingency table: {str(e)}")
        
        return analysis
    
    def _interpret_cramers_v(self, v: float) -> str:
        """
        Interpret Cramér's V value
        
        Args:
            v: Cramér's V value
            
        Returns:
            String interpretation
        """
        if v < 0.1:
            return "negligible"
        elif v < 0.3:
            return "weak"
        elif v < 0.5:
            return "moderate"
        else:
            return "strong"
    
    def suggest_variable_pairs(self, df: pd.DataFrame, categorical_columns: List[str]) -> List[Dict[str, Any]]:
        """
        Suggest good variable pairs for contingency tables
        
        Args:
            df: DataFrame
            categorical_columns: List of categorical column names
            
        Returns:
            List of suggested pairs with scores
        """
        suggestions = []
        
        # Analyze all possible pairs
        for i, var1 in enumerate(categorical_columns):
            for var2 in categorical_columns[i+1:]:
                pair_analysis = self.analyze_pair(df, var1, var2)
                
                if pair_analysis['is_valid']:
                    # Calculate suitability score
                    score = self._calculate_pair_score(pair_analysis)
                    
                    suggestions.append({
                        'row_var': var1,
                        'col_var': var2,
                        'score': score,
                        'completeness': pair_analysis['completeness'],
                        'table_shape': pair_analysis['table_shape'],
                        'warnings': pair_analysis['warnings'],
                        'association_strength': pair_analysis.get('association_preview', {}).get('strength', 'unknown')
                    })
        
        # Sort by score
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        
        return suggestions
    
    def _calculate_pair_score(self, analysis: Dict[str, Any]) -> float:
        """
        Calculate suitability score for a variable pair
        
        Args:
            analysis: Pair analysis dictionary
            
        Returns:
            Score between 0 and 100
        """
        score = 100.0
        
        # Penalize for missing data
        completeness_penalty = (100 - analysis['completeness']) * 0.3
        score -= completeness_penalty
        
        # Penalize for sparsity
        if 'sparse_percentage' in analysis:
            sparsity_penalty = analysis['sparse_percentage'] * 0.5
            score -= sparsity_penalty
        
        # Penalize for too many or too few categories
        if analysis['table_shape']:
            rows, cols = analysis['table_shape']
            
            # Ideal is between 3-10 categories per variable
            if rows < 3:
                score -= 10
            elif rows > 10:
                score -= (rows - 10) * 2
            
            if cols < 3:
                score -= 10
            elif cols > 10:
                score -= (cols - 10) * 2
        
        # Penalize for warnings
        score -= len(analysis.get('warnings', [])) * 5
        
        # Bonus for moderate to strong association
        if 'association_preview' in analysis:
            strength = analysis['association_preview'].get('strength', '')
            if strength == 'moderate':
                score += 5
            elif strength == 'strong':
                score += 10
        
        return max(0, min(100, score))
    
    def validate_selection(self, df: pd.DataFrame, row_var: str, col_var: str) -> Tuple[bool, List[str]]:
        """
        Validate selected variables for contingency table
        
        Args:
            df: DataFrame
            row_var: Row variable name
            col_var: Column variable name
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check existence
        if row_var not in df.columns:
            errors.append(f"Row variable '{row_var}' not found")
        
        if col_var not in df.columns:
            errors.append(f"Column variable '{col_var}' not found")
        
        if errors:
            return False, errors
        
        # Check data types
        row_dtype = df[row_var].dtype
        col_dtype = df[col_var].dtype
        
        if pd.api.types.is_numeric_dtype(row_dtype) and df[row_var].nunique() > self.max_categories:
            errors.append(f"Row variable '{row_var}' appears to be continuous. Consider categorizing it first.")
        
        if pd.api.types.is_numeric_dtype(col_dtype) and df[col_var].nunique() > self.max_categories:
            errors.append(f"Column variable '{col_var}' appears to be continuous. Consider categorizing it first.")
        
        # Check for sufficient data
        complete_cases = df[[row_var, col_var]].notna().all(axis=1).sum()
        if complete_cases < 10:
            errors.append(f"Insufficient complete cases ({complete_cases} < 10)")
        
        return len(errors) == 0, errors