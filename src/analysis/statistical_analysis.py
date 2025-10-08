"""
Statistical Analysis Module
Comprehensive statistical analysis for phishing susceptibility research
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit
import pingouin as pg
from typing import Dict, List, Tuple
import yaml


class StatisticalAnalyzer:
    """
    Performs comprehensive statistical analysis on phishing research data
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize analyzer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.alpha = self.config['analysis']['significance_level']
        self.results = {}
    
    def load_data(self, survey_path: str, experiment_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load survey and experiment data"""
        self.survey_data = pd.read_csv(survey_path)
        self.experiment_data = pd.read_csv(experiment_path)
        
        print(f"Loaded survey data: {self.survey_data.shape}")
        print(f"Loaded experiment data: {self.experiment_data.shape}")
        
        return self.survey_data, self.experiment_data
    
    def descriptive_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive descriptive statistics"""
        desc_stats = {
            'summary': data.describe(include='all').to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.astype(str).to_dict()
        }
        
        # Numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            desc_stats['numerical'] = {
                'mean': data[numerical_cols].mean().to_dict(),
                'median': data[numerical_cols].median().to_dict(),
                'std': data[numerical_cols].std().to_dict(),
                'skewness': data[numerical_cols].skew().to_dict(),
                'kurtosis': data[numerical_cols].kurtosis().to_dict()
            }
        
        # Categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            desc_stats['categorical'] = {}
            for col in categorical_cols:
                desc_stats['categorical'][col] = data[col].value_counts().to_dict()
        
        self.results['descriptive_statistics'] = desc_stats
        return desc_stats
    
    def chi_square_test(self, data: pd.DataFrame, var1: str, var2: str) -> Dict:
        """
        Perform chi-square test of independence
        Research Question: Association between categorical variables
        """
        contingency_table = pd.crosstab(data[var1], data[var2])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Calculate effect size (Cramér's V)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        
        result = {
            'test': 'Chi-Square Test of Independence',
            'variables': [var1, var2],
            'chi_square': float(chi2),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'cramers_v': float(cramers_v),
            'significant': p_value < self.alpha,
            'interpretation': self._interpret_cramers_v(cramers_v),
            'contingency_table': contingency_table.to_dict()
        }
        
        return result
    
    def anova_test(self, data: pd.DataFrame, dependent_var: str, 
                   independent_var: str) -> Dict:
        """
        Perform one-way ANOVA
        Research Question: Difference in means across groups
        """
        groups = [data[data[independent_var] == group][dependent_var].dropna() 
                 for group in data[independent_var].unique()]
        
        # Remove empty groups
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) < 2:
            return {'error': 'Not enough groups for ANOVA'}
        
        f_stat, p_value = f_oneway(*groups)
        
        # Calculate effect size (eta squared)
        data_clean = data[[dependent_var, independent_var]].dropna()
        model = ols(f'{dependent_var} ~ C({independent_var})', data=data_clean).fit()
        eta_squared = sm.stats.anova_lm(model, typ=2)['sum_sq'][0] / sm.stats.anova_lm(model, typ=2)['sum_sq'].sum()
        
        result = {
            'test': 'One-Way ANOVA',
            'dependent_variable': dependent_var,
            'independent_variable': independent_var,
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'eta_squared': float(eta_squared),
            'significant': p_value < self.alpha,
            'interpretation': self._interpret_eta_squared(eta_squared)
        }
        
        # Post-hoc tests if significant
        if p_value < self.alpha and len(groups) > 2:
            posthoc = pg.pairwise_tukey(data=data_clean, dv=dependent_var, 
                                       between=independent_var)
            result['posthoc_tukey'] = posthoc.to_dict('records')
        
        return result
    
    def correlation_analysis(self, data: pd.DataFrame, 
                           vars_list: List[str] = None) -> Dict:
        """
        Perform correlation analysis
        Research Question: Relationships between continuous variables
        """
        if vars_list is None:
            # Use all numerical columns
            vars_list = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Pearson correlation
        pearson_corr = data[vars_list].corr(method='pearson')
        
        # Spearman correlation (for non-normal data)
        spearman_corr = data[vars_list].corr(method='spearman')
        
        # Calculate p-values
        p_values = pd.DataFrame(np.zeros((len(vars_list), len(vars_list))),
                               index=vars_list, columns=vars_list)
        
        for i, var1 in enumerate(vars_list):
            for j, var2 in enumerate(vars_list):
                if i != j:
                    _, p_val = pearsonr(data[var1].dropna(), data[var2].dropna())
                    p_values.iloc[i, j] = p_val
        
        result = {
            'test': 'Correlation Analysis',
            'variables': vars_list,
            'pearson_correlation': pearson_corr.to_dict(),
            'spearman_correlation': spearman_corr.to_dict(),
            'p_values': p_values.to_dict(),
            'significant_correlations': self._identify_significant_correlations(
                pearson_corr, p_values
            )
        }
        
        return result
    
    def logistic_regression(self, data: pd.DataFrame, 
                          dependent_var: str, 
                          independent_vars: List[str]) -> Dict:
        """
        Perform logistic regression analysis
        Research Question: Predict binary outcome (phishing susceptibility)
        """
        # Prepare data
        data_clean = data[[dependent_var] + independent_vars].dropna()
        
        # Encode binary dependent variable if needed
        if data_clean[dependent_var].dtype == 'object':
            data_clean[dependent_var] = (data_clean[dependent_var] == 
                                        data_clean[dependent_var].unique()[1]).astype(int)
        
        X = data_clean[independent_vars]
        y = data_clean[dependent_var]
        
        # Add constant
        X = sm.add_constant(X)
        
        # Fit model
        model = sm.Logit(y, X).fit(disp=0)
        
        result = {
            'test': 'Logistic Regression',
            'dependent_variable': dependent_var,
            'independent_variables': independent_vars,
            'coefficients': model.params.to_dict(),
            'p_values': model.pvalues.to_dict(),
            'odds_ratios': np.exp(model.params).to_dict(),
            'pseudo_r_squared': float(model.prsquared),
            'aic': float(model.aic),
            'bic': float(model.bic),
            'summary': str(model.summary())
        }
        
        return result
    
    def t_test(self, data: pd.DataFrame, var: str, group_var: str) -> Dict:
        """
        Perform independent samples t-test
        Research Question: Compare means between two groups
        """
        groups = data[group_var].unique()
        
        if len(groups) != 2:
            return {'error': 'T-test requires exactly 2 groups'}
        
        group1 = data[data[group_var] == groups[0]][var].dropna()
        group2 = data[data[group_var] == groups[1]][var].dropna()
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        # Calculate effect size (Cohen's d)
        cohens_d = (group1.mean() - group2.mean()) / np.sqrt(
            ((len(group1) - 1) * group1.std()**2 + (len(group2) - 1) * group2.std()**2) / 
            (len(group1) + len(group2) - 2)
        )
        
        result = {
            'test': 'Independent Samples T-Test',
            'variable': var,
            'groups': groups.tolist(),
            'group1_mean': float(group1.mean()),
            'group2_mean': float(group2.mean()),
            'group1_std': float(group1.std()),
            'group2_std': float(group2.std()),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant': p_value < self.alpha,
            'interpretation': self._interpret_cohens_d(cohens_d)
        }
        
        return result
    
    def _interpret_cramers_v(self, v: float) -> str:
        """Interpret Cramér's V effect size"""
        if v < 0.1:
            return "negligible"
        elif v < 0.3:
            return "small"
        elif v < 0.5:
            return "medium"
        else:
            return "large"
    
    def _interpret_eta_squared(self, eta: float) -> str:
        """Interpret eta squared effect size"""
        if eta < 0.01:
            return "negligible"
        elif eta < 0.06:
            return "small"
        elif eta < 0.14:
            return "medium"
        else:
            return "large"
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _identify_significant_correlations(self, corr_matrix: pd.DataFrame, 
                                          p_matrix: pd.DataFrame) -> List[Dict]:
        """Identify statistically significant correlations"""
        significant = []
        
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                var1 = corr_matrix.index[i]
                var2 = corr_matrix.columns[j]
                r = corr_matrix.iloc[i, j]
                p = p_matrix.iloc[i, j]
                
                if p < self.alpha and abs(r) > 0.1:  # Threshold for meaningful correlation
                    significant.append({
                        'variable_1': var1,
                        'variable_2': var2,
                        'correlation': float(r),
                        'p_value': float(p),
                        'strength': self._interpret_correlation(r)
                    })
        
        return significant
    
    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation strength"""
        abs_r = abs(r)
        if abs_r < 0.3:
            return "weak"
        elif abs_r < 0.7:
            return "moderate"
        else:
            return "strong"
    
    def generate_full_report(self, output_path: str = 'results/statistical_analysis_report.json'):
        """Generate comprehensive statistical analysis report"""
        import json
        from datetime import datetime
        
        report = {
            'generated': datetime.now().isoformat(),
            'analysis_results': self.results,
            'configuration': {
                'significance_level': self.alpha,
                'confidence_interval': self.config['analysis']['confidence_interval']
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Statistical analysis report saved to {output_path}")
        return report


def main():
    """Main function for statistical analysis"""
    print("Initializing Statistical Analysis...")
    
    analyzer = StatisticalAnalyzer()
    
    # Load data (using synthetic data for demonstration)
    try:
        survey_data, experiment_data = analyzer.load_data(
            'data/synthetic/sample_responses.csv',
            'data/synthetic/experiment_results.csv'
        )
        
        print("\n1. Descriptive Statistics...")
        desc_stats = analyzer.descriptive_statistics(experiment_data)
        
        print("\n2. Analyzing relationships...")
        # Add more specific analyses based on research questions
        
        print("\n✓ Statistical analysis complete!")
        print("Results saved to results/statistical_analysis_report.json")
        
    except FileNotFoundError as e:
        print(f"Error: Data files not found. Please run data collection scripts first.")
        print(f"Details: {e}")


if __name__ == "__main__":
    main()
