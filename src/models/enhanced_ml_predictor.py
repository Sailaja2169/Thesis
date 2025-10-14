"""
Enhanced Machine Learning Module with Explainability
Includes: SHAP values, Ensemble Stacking, Uncertainty Quantification
For IEEE publication-quality ML analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class EnhancedVulnerabilityPredictor:
    """
    Advanced ML for phishing vulnerability prediction with explainability
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize predictor"""
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models = {}
        self.stacking_model = None
        self.best_model = None
        self.scaler = StandardScaler()
        self.shap_explainer = None
        self.shap_values = None
    
    # =================================================================
    # ENSEMBLE STACKING (Better than individual models)
    # =================================================================
    
    def train_stacking_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict:
        """
        Train ensemble that combines predictions from multiple models
        
        Why better: Leverages strengths of different algorithms
        Expected: 2-5% accuracy improvement over best single model
        
        Architecture:
            Level 0 (Base Models):
                - Random Forest
                - Gradient Boosting
                - SVM
            Level 1 (Meta-learner):
                - Logistic Regression (combines base predictions)
        
        Args:
            X_train: Training features
            y_train: Training labels
        
        Returns:
            Dict with training results
        """
        print("\n🚀 Training Stacking Ensemble...")
        
        # Handle class imbalance
        smote = SMOTE(random_state=self.config['ml_models']['random_state'])
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_balanced)
        
        # Define base models
        base_models = [
            ('rf', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.config['ml_models']['random_state'],
                n_jobs=-1
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.config['ml_models']['random_state']
            )),
            ('svm', SVC(
                kernel='rbf',
                probability=True,
                random_state=self.config['ml_models']['random_state']
            ))
        ]
        
        # Meta-learner
        meta_learner = LogisticRegression(
            max_iter=1000,
            random_state=self.config['ml_models']['random_state']
        )
        
        # Create stacking ensemble
        self.stacking_model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=self.config['ml_models']['cross_validation_folds'],
            n_jobs=-1
        )
        
        # Train ensemble
        self.stacking_model.fit(X_scaled, y_balanced)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.stacking_model,
            X_scaled,
            y_balanced,
            cv=self.config['ml_models']['cross_validation_folds'],
            scoring='roc_auc'
        )
        
        results = {
            'model_type': 'Stacking Ensemble',
            'base_models': ['Random Forest', 'Gradient Boosting', 'SVM'],
            'meta_learner': 'Logistic Regression',
            'cv_mean_roc_auc': float(cv_scores.mean()),
            'cv_std_roc_auc': float(cv_scores.std()),
            'training_samples': len(X_balanced)
        }
        
        print(f"✓ Stacking Ensemble CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return results
    
    # =================================================================
    # SHAP EXPLAINABILITY (Better than feature importance)
    # =================================================================
    
    def calculate_shap_values(
        self,
        model,
        X_test: pd.DataFrame,
        feature_names: List[str],
        sample_size: int = 100
    ) -> Dict:
        """
        Calculate SHAP values for model interpretability
        
        SHAP (SHapley Additive exPlanations) provides:
        - Individual prediction explanations
        - Feature importance with directionality
        - Interaction effects
        
        Better than feature_importance_ because:
        - Shows contribution to EACH prediction (not just overall)
        - Accounts for feature interactions
        - Theoretically grounded (game theory)
        
        Args:
            model: Trained model
            X_test: Test data
            feature_names: List of feature names
            sample_size: Number of samples for explanation (computational cost)
        
        Returns:
            Dict with SHAP values and explanations
        """
        try:
            import shap
        except ImportError:
            print("⚠️ SHAP not installed. Run: pip install shap")
            return {'error': 'SHAP library not available'}
        
        print("\n🔍 Calculating SHAP values for explainability...")
        
        # Sample data for computational efficiency
        if len(X_test) > sample_size:
            sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
            X_sample = X_test.iloc[sample_indices]
        else:
            X_sample = X_test
        
        # Scale features
        X_scaled = self.scaler.transform(X_sample)
        
        # Create SHAP explainer (TreeExplainer for tree-based models)
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            
            # For binary classification, extract positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class
            
        except Exception:
            # Fallback to KernelExplainer (slower but works for any model)
            explainer = shap.KernelExplainer(
                model.predict_proba,
                shap.sample(X_scaled, 50)  # Background dataset
            )
            shap_values = explainer.shap_values(X_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        
        self.shap_explainer = explainer
        self.shap_values = shap_values
        
        # Calculate mean absolute SHAP values (feature importance)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create feature importance dataframe
        shap_importance = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        # Calculate for each feature: percentage contribution
        total_shap = mean_abs_shap.sum()
        shap_importance['importance_pct'] = (shap_importance['mean_abs_shap'] / total_shap) * 100
        
        results = {
            'feature_importance': shap_importance.to_dict('records'),
            'top_5_features': shap_importance.head(5)['feature'].tolist(),
            'shap_values_computed': True,
            'n_samples_explained': len(X_sample)
        }
        
        print(f"✓ SHAP values calculated for {len(X_sample)} samples")
        print(f"\nTop 5 Most Important Features:")
        for i, row in shap_importance.head(5).iterrows():
            print(f"  {i+1}. {row['feature']}: {row['importance_pct']:.2f}%")
        
        return results
    
    def explain_individual_prediction(
        self,
        individual_features: Dict,
        feature_names: List[str]
    ) -> Dict:
        """
        Explain why a specific individual was classified as high/low risk
        
        Example output:
        "This person is HIGH RISK because:
         - High authority bias (+0.23 to risk)
         - High stress level (+0.18 to risk)
         - No training (-0.15 protection)
         - Fast response time (+0.12 to risk)"
        
        Args:
            individual_features: Dict of feature values for one person
            feature_names: List of feature names
        
        Returns:
            Dict with explanation
        """
        if self.shap_explainer is None:
            return {'error': 'Must call calculate_shap_values first'}
        
        # Convert to array
        X_individual = pd.DataFrame([individual_features])
        X_scaled = self.scaler.transform(X_individual)
        
        # Get SHAP values for this individual
        shap_vals = self.shap_explainer.shap_values(X_scaled)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1][0]  # Positive class, first sample
        else:
            shap_vals = shap_vals[0]
        
        # Get prediction
        prediction = self.best_model.predict(X_scaled)[0]
        probability = self.best_model.predict_proba(X_scaled)[0][1]
        
        # Create explanation
        feature_contributions = pd.DataFrame({
            'feature': feature_names,
            'value': [individual_features[f] for f in feature_names],
            'shap_value': shap_vals
        }).sort_values('shap_value', key=abs, ascending=False)
        
        explanation = {
            'prediction': 'HIGH RISK' if prediction == 1 else 'LOW RISK',
            'probability': float(probability),
            'top_risk_factors': [],
            'top_protective_factors': [],
            'all_contributions': feature_contributions.to_dict('records')
        }
        
        # Extract top contributors
        for _, row in feature_contributions.iterrows():
            contribution = {
                'feature': row['feature'],
                'value': float(row['value']),
                'shap_contribution': float(row['shap_value'])
            }
            
            if row['shap_value'] > 0.01:  # Increases risk
                explanation['top_risk_factors'].append(contribution)
            elif row['shap_value'] < -0.01:  # Decreases risk (protective)
                explanation['top_protective_factors'].append(contribution)
        
        return explanation
    
    # =================================================================
    # UNCERTAINTY QUANTIFICATION
    # =================================================================
    
    def predict_with_uncertainty(
        self,
        features: Dict,
        n_estimators: int = None
    ) -> Dict:
        """
        Provide prediction with confidence intervals
        
        Instead of just "HIGH RISK", output:
        "HIGH RISK with 85% confidence (95% CI: 0.72-0.93)"
        
        Method: Use Random Forest's ensemble predictions for uncertainty
        
        Args:
            features: Individual feature values
            n_estimators: Number of trees (if None, use model default)
        
        Returns:
            Dict with prediction and uncertainty estimates
        """
        # Convert to DataFrame
        X = pd.DataFrame([features])
        X_scaled = self.scaler.transform(X)
        
        if isinstance(self.best_model, RandomForestClassifier):
            # Get predictions from each tree
            tree_predictions = []
            for tree in self.best_model.estimators_:
                pred = tree.predict_proba(X_scaled)[0][1]  # Probability of positive class
                tree_predictions.append(pred)
            
            tree_predictions = np.array(tree_predictions)
            
            # Calculate statistics
            mean_pred = tree_predictions.mean()
            std_pred = tree_predictions.std()
            
            # 95% confidence interval
            ci_lower = np.percentile(tree_predictions, 2.5)
            ci_upper = np.percentile(tree_predictions, 97.5)
            
            # Prediction certainty (inverse of std)
            certainty = 1 - min(std_pred * 2, 1.0)  # Scale to 0-1
            
            result = {
                'prediction': 'HIGH RISK' if mean_pred > 0.5 else 'LOW RISK',
                'probability': float(mean_pred),
                'uncertainty': float(std_pred),
                'certainty': float(certainty),
                'confidence_interval_95': (float(ci_lower), float(ci_upper)),
                'interpretation': self._interpret_uncertainty(mean_pred, std_pred, certainty)
            }
        else:
            # For non-RF models, use single prediction
            prob = self.best_model.predict_proba(X_scaled)[0][1]
            result = {
                'prediction': 'HIGH RISK' if prob > 0.5 else 'LOW RISK',
                'probability': float(prob),
                'uncertainty': 'N/A (model does not support uncertainty quantification)',
                'note': 'Use Random Forest for uncertainty estimates'
            }
        
        return result
    
    def _interpret_uncertainty(self, mean: float, std: float, certainty: float) -> str:
        """Interpret uncertainty for practitioners"""
        if certainty > 0.9:
            confidence_label = "very high confidence"
        elif certainty > 0.7:
            confidence_label = "high confidence"
        elif certainty > 0.5:
            confidence_label = "moderate confidence"
        else:
            confidence_label = "low confidence"
        
        interpretation = f"""
        Prediction: {'HIGH RISK' if mean > 0.5 else 'LOW RISK'}
        Probability: {mean:.2%}
        Confidence: {confidence_label} (certainty={certainty:.2%})
        
        This means the model is {confidence_label.split()[0]} {confidence_label.split()[1]}
        in its prediction. {'Consider additional assessment.' if certainty < 0.7 else 'Prediction is reliable.'}
        """
        
        return interpretation.strip()
    
    # =================================================================
    # MODEL COMPARISON & SELECTION
    # =================================================================
    
    def compare_all_models(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Compare all models for IEEE publication table
        
        Returns:
            DataFrame with model comparison metrics
        """
        results = []
        
        models_to_compare = {
            'Random Forest': self.models.get('random_forest'),
            'Gradient Boosting': self.models.get('gradient_boosting'),
            'Logistic Regression': self.models.get('logistic_regression'),
            'SVM': self.models.get('svm'),
            'Stacking Ensemble': self.stacking_model
        }
        
        X_scaled = self.scaler.transform(X_test)
        
        for name, model in models_to_compare.items():
            if model is None:
                continue
            
            y_pred = model.predict(X_scaled)
            y_pred_proba = model.predict_proba(X_scaled)[:, 1]
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            results.append({
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1-Score': f1_score(y_test, y_pred, zero_division=0),
                'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
            })
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.round(4)
        
        return comparison_df


def main():
    """Example usage"""
    print("="*60)
    print("ENHANCED ML MODULE WITH EXPLAINABILITY")
    print("IEEE Publication-Quality Machine Learning")
    print("="*60)
    
    print("\n✓ Module includes:")
    print("  - Stacking Ensemble (better accuracy)")
    print("  - SHAP Explainability (interpretable predictions)")
    print("  - Uncertainty Quantification (confidence intervals)")
    print("  - Individual Prediction Explanations")
    print("  - Model Comparison Tables")
    
    print("\n📝 Ready for use with real data!")


if __name__ == "__main__":
    main()
