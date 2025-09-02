"""
Recommendation Engine Module
Provides intelligent recommendations for variable selection
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from itertools import combinations


class RecommendationEngine:
    """Engine for providing intelligent variable recommendations"""
    
    def __init__(self):
        """Initialize the recommendation engine"""
        self.max_recommendations = 10
        self.min_association_strength = 0.1
        self.ideal_categories_range = (3, 10)
        self.min_completeness = 80  # Minimum % of complete cases
    
    def get_recommendations(self, 
                           df: pd.DataFrame, 
                           schema: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get variable pair recommendations for contingency tables
        
        Args:
            df: DataFrame
            schema: Schema dictionary from SchemaDetector
            
        Returns:
            List of recommended variable pairs with scores
        """
        recommendations = []
        
        # Get categorical variables
        categorical_vars = [col for col, info in schema.items() 
                          if info['data_type'] in ['categorical', 'boolean']]
        
        if len(categorical_vars) < 2:
            return []
        
        # Analyze all possible pairs
        for var1, var2 in combinations(categorical_vars, 2):
            score, reasons = self._score_pair(df, var1, var2, schema)
            
            if score > 0:
                recommendations.append({
                    'row_var': var1,
                    'col_var': var2,
                    'score': score,
                    'reasons': reasons,
                    'preview': self._get_pair_preview(df, var1, var2)
                })
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Add rank and format reasons
        for i, rec in enumerate(recommendations[:self.max_recommendations], 1):
            rec['rank'] = i
            rec['recommendation_text'] = self._format_recommendation(rec)
        
        return recommendations[:self.max_recommendations]
    
    def _score_pair(self, 
                   df: pd.DataFrame, 
                   var1: str, 
                   var2: str, 
                   schema: Dict[str, Dict[str, Any]]) -> Tuple[float, List[str]]:
        """
        Score a variable pair for suitability
        
        Args:
            df: DataFrame
            var1: First variable name
            var2: Second variable name
            schema: Schema dictionary
            
        Returns:
            Tuple of (score, list of scoring reasons)
        """
        score = 0
        reasons = []
        
        # Check completeness
        complete_cases = df[[var1, var2]].notna().all(axis=1).sum()
        completeness = (complete_cases / len(df)) * 100
        
        if completeness < 50:
            return 0, ["Too many missing values"]
        
        if completeness >= self.min_completeness:
            score += 20
            reasons.append(f"High completeness ({completeness:.1f}%)")
        else:
            score += 10
        
        # Check cardinality
        var1_unique = df[var1].nunique()
        var2_unique = df[var2].nunique()
        
        # Ideal cardinality scoring
        for unique_count in [var1_unique, var2_unique]:
            if self.ideal_categories_range[0] <= unique_count <= self.ideal_categories_range[1]:
                score += 15
                reasons.append(f"Ideal number of categories ({unique_count})")
            elif unique_count < self.ideal_categories_range[0]:
                score += 5
                reasons.append(f"Few categories ({unique_count})")
            elif unique_count <= 20:
                score += 10
                reasons.append(f"Moderate categories ({unique_count})")
            else:
                score += 0  # Too many categories
        
        # Check for semantic relationships
        semantic_score = self._check_semantic_relationship(var1, var2)
        if semantic_score > 0:
            score += semantic_score
            reasons.append("Likely semantic relationship")
        
        # Quick association check (if not too many categories)
        if var1_unique <= 20 and var2_unique <= 20 and complete_cases >= 30:
            try:
                ct = pd.crosstab(df[var1], df[var2])
                chi2 = self._quick_chi2(ct)
                n = ct.sum().sum()
                min_dim = min(ct.shape[0] - 1, ct.shape[1] - 1)
                cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
                
                if cramers_v >= 0.3:
                    score += 25
                    reasons.append(f"Strong association detected (V={cramers_v:.2f})")
                elif cramers_v >= 0.2:
                    score += 15
                    reasons.append(f"Moderate association detected (V={cramers_v:.2f})")
                elif cramers_v >= self.min_association_strength:
                    score += 10
                    reasons.append(f"Weak association detected (V={cramers_v:.2f})")
            except:
                pass
        
        # Check for balanced distribution
        for var in [var1, var2]:
            value_counts = df[var].value_counts()
            if len(value_counts) > 1:
                imbalance = value_counts.max() / value_counts.min()
                if imbalance < 5:
                    score += 5
                    reasons.append(f"Balanced distribution in {var}")
        
        # Domain-specific scoring
        domain_score = self._check_domain_relevance(var1, var2)
        if domain_score > 0:
            score += domain_score
            reasons.append("Domain-relevant pairing")
        
        return score, reasons
    
    def _quick_chi2(self, ct: pd.DataFrame) -> float:
        """
        Quick chi-square calculation without full scipy stats
        
        Args:
            ct: Contingency table
            
        Returns:
            Chi-square statistic
        """
        row_totals = ct.sum(axis=1)
        col_totals = ct.sum(axis=0)
        total = ct.sum().sum()
        
        expected = np.outer(row_totals, col_totals) / total
        chi2 = ((ct - expected) ** 2 / expected).sum().sum()
        
        return chi2
    
    def _check_semantic_relationship(self, var1: str, var2: str) -> float:
        """
        Check for semantic relationships between variable names
        
        Args:
            var1: First variable name
            var2: Second variable name
            
        Returns:
            Semantic relationship score
        """
        score = 0
        
        # Common semantic pairs
        semantic_pairs = [
            ('age', 'income'),
            ('education', 'income'),
            ('education', 'employment'),
            ('gender', 'income'),
            ('region', 'preference'),
            ('category', 'satisfaction'),
            ('product', 'rating'),
            ('treatment', 'outcome'),
            ('diagnosis', 'treatment'),
            ('before', 'after'),
            ('control', 'treatment'),
            ('month', 'sales'),
            ('quarter', 'revenue'),
            ('department', 'satisfaction'),
            ('experience', 'salary')
        ]
        
        var1_lower = var1.lower()
        var2_lower = var2.lower()
        
        for pair in semantic_pairs:
            if (pair[0] in var1_lower and pair[1] in var2_lower) or \
               (pair[1] in var1_lower and pair[0] in var2_lower):
                score += 10
                break
        
        # Check for time-based relationships
        time_keywords = ['year', 'month', 'quarter', 'week', 'day', 'date', 'time', 'period']
        if any(keyword in var1_lower for keyword in time_keywords) and \
           any(keyword not in var2_lower for keyword in time_keywords):
            score += 5
        
        # Check for demographic relationships
        demographic_keywords = ['age', 'gender', 'sex', 'race', 'ethnicity', 'education', 'income']
        outcome_keywords = ['outcome', 'result', 'satisfaction', 'preference', 'choice', 'rating']
        
        var1_is_demographic = any(keyword in var1_lower for keyword in demographic_keywords)
        var2_is_demographic = any(keyword in var2_lower for keyword in demographic_keywords)
        var1_is_outcome = any(keyword in var1_lower for keyword in outcome_keywords)
        var2_is_outcome = any(keyword in var2_lower for keyword in outcome_keywords)
        
        if (var1_is_demographic and var2_is_outcome) or (var2_is_demographic and var1_is_outcome):
            score += 8
        
        return score
    
    def _check_domain_relevance(self, var1: str, var2: str) -> float:
        """
        Check domain-specific relevance of variable pairing
        
        Args:
            var1: First variable name
            var2: Second variable name
            
        Returns:
            Domain relevance score
        """
        score = 0
        
        # Medical/Healthcare domain
        medical_terms = ['diagnosis', 'treatment', 'symptom', 'patient', 'medication', 
                        'outcome', 'condition', 'therapy', 'disease']
        
        # Business/Sales domain
        business_terms = ['sales', 'revenue', 'customer', 'product', 'region', 
                         'quarter', 'segment', 'channel', 'category']
        
        # Survey/Research domain
        survey_terms = ['satisfaction', 'rating', 'preference', 'opinion', 'response',
                       'agree', 'disagree', 'likelihood', 'recommend']
        
        var1_lower = var1.lower()
        var2_lower = var2.lower()
        
        # Check if both variables are from the same domain
        for domain_terms in [medical_terms, business_terms, survey_terms]:
            var1_match = any(term in var1_lower for term in domain_terms)
            var2_match = any(term in var2_lower for term in domain_terms)
            
            if var1_match and var2_match:
                score += 5
                break
        
        return score
    
    def _get_pair_preview(self, df: pd.DataFrame, var1: str, var2: str) -> Dict[str, Any]:
        """
        Get a preview of the contingency table for a pair
        
        Args:
            df: DataFrame
            var1: First variable
            var2: Second variable
            
        Returns:
            Preview dictionary
        """
        preview = {}
        
        try:
            # Create small contingency table
            ct = pd.crosstab(df[var1], df[var2])
            
            # Limit size for preview
            if ct.shape[0] > 5:
                ct = ct.iloc[:5, :]
            if ct.shape[1] > 5:
                ct = ct.iloc[:, :5]
            
            preview['table_shape'] = f"{ct.shape[0]} × {ct.shape[1]}"
            preview['total_cells'] = ct.shape[0] * ct.shape[1]
            preview['non_zero_cells'] = (ct > 0).sum().sum()
            preview['sample_size'] = ct.sum().sum()
            
        except:
            preview['error'] = "Could not generate preview"
        
        return preview
    
    def _format_recommendation(self, rec: Dict[str, Any]) -> str:
        """
        Format recommendation for display
        
        Args:
            rec: Recommendation dictionary
            
        Returns:
            Formatted recommendation text
        """
        text = f"**Recommendation #{rec['rank']}**: "
        text += f"`{rec['row_var']}` × `{rec['col_var']}`\n"
        text += f"Score: {rec['score']:.1f}/100\n"
        
        if rec['reasons']:
            text += "Reasons: " + ", ".join(rec['reasons'][:3])
        
        if 'preview' in rec and 'table_shape' in rec['preview']:
            text += f"\nTable size: {rec['preview']['table_shape']}"
        
        return text
    
    def get_categorization_recommendations(self, 
                                          df: pd.DataFrame, 
                                          column: str) -> Dict[str, Any]:
        """
        Get recommendations for categorizing a continuous variable
        
        Args:
            df: DataFrame
            column: Column name to categorize
            
        Returns:
            Dictionary with categorization recommendations
        """
        recommendations = {
            'column': column,
            'strategies': []
        }
        
        series = df[column].dropna()
        
        if len(series) == 0:
            return recommendations
        
        # Analyze distribution
        mean = series.mean()
        median = series.median()
        std = series.std()
        skewness = series.skew()
        
        # Equal width binning
        n_bins = self._suggest_n_bins(len(series))
        recommendations['strategies'].append({
            'method': 'equal_width',
            'n_bins': n_bins,
            'description': 'Divides the range into equal-width intervals',
            'best_for': 'Uniformly distributed data'
        })
        
        # Equal frequency binning
        recommendations['strategies'].append({
            'method': 'equal_frequency',
            'n_bins': n_bins,
            'description': 'Creates bins with approximately equal number of observations',
            'best_for': 'Skewed distributions'
        })
        
        # Quartiles
        recommendations['strategies'].append({
            'method': 'quartiles',
            'n_bins': 4,
            'description': 'Divides data into quartiles (Q1, Q2, Q3, Q4)',
            'best_for': 'Standard statistical analysis'
        })
        
        # Based on standard deviations (if normally distributed)
        if abs(skewness) < 0.5:  # Approximately normal
            recommendations['strategies'].append({
                'method': 'std_dev',
                'bins': [mean - 2*std, mean - std, mean, mean + std, mean + 2*std],
                'description': 'Bins based on standard deviations from mean',
                'best_for': 'Normally distributed data'
            })
        
        # Domain-specific recommendations
        if 'age' in column.lower():
            recommendations['strategies'].append({
                'method': 'domain_specific',
                'bins': [0, 18, 35, 50, 65, 100],
                'labels': ['<18', '18-34', '35-49', '50-64', '65+'],
                'description': 'Standard age groups',
                'best_for': 'Demographic analysis'
            })
        
        elif 'income' in column.lower() or 'salary' in column.lower():
            # Determine scale
            if series.max() > 10000:  # Likely annual income
                recommendations['strategies'].append({
                    'method': 'domain_specific',
                    'bins': [0, 25000, 50000, 75000, 100000, float('inf')],
                    'labels': ['<25k', '25-50k', '50-75k', '75-100k', '>100k'],
                    'description': 'Standard income brackets',
                    'best_for': 'Income analysis'
                })
        
        return recommendations
    
    def _suggest_n_bins(self, n: int) -> int:
        """
        Suggest number of bins based on sample size
        
        Args:
            n: Sample size
            
        Returns:
            Suggested number of bins
        """
        # Sturges' rule
        sturges = int(np.ceil(np.log2(n) + 1))
        
        # Square root rule
        sqrt_rule = int(np.ceil(np.sqrt(n)))
        
        # Rice's rule
        rice = int(np.ceil(2 * np.cbrt(n)))
        
        # Take median and cap at reasonable limits
        suggested = int(np.median([sturges, sqrt_rule, rice]))
        suggested = max(3, min(suggested, 10))
        
        return suggested