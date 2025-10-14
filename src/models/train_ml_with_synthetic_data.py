"""
Train ML Models with Synthetic Data
Trains vulnerability prediction models on 400+ synthetic responses
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
import json

# Create output directory
output_dir = Path('models/saved')
output_dir.mkdir(parents=True, exist_ok=True)


def load_processed_data():
    """Load processed synthetic data"""
    print("📂 Loading processed synthetic data...")
    
    df = pd.read_csv('data/processed/synthetic_survey_data_cleaned.csv')
    print(f"  ✓ Loaded {len(df)} participants")
    
    return df


def prepare_features(df):
    """Prepare features for ML training"""
    print("\n🔧 Preparing features...")
    
    # Select relevant features
    feature_cols = [
        'Age bracket',
        'Years of professional computer use',
        'Prior cybersecurity training?',
        'On a scale of 1-5, how important is data privacy to you in your professional role?',
        'How confident are you in identifying a phishing email?',
        'How would you rate your overall digital/computer literacy?',
        'I feel stressed at work most days.',
        'I tend to follow requests that appear to come from managers.',
        'Multitasking frequency'
    ]
    
    df_features = df[feature_cols].copy()
    
    # Encode categorical variables
    le_age = LabelEncoder()
    le_exp = LabelEncoder()
    le_multitask = LabelEncoder()
    
    df_features['age_encoded'] = le_age.fit_transform(df['Age bracket'])
    df_features['experience_encoded'] = le_exp.fit_transform(df['Years of professional computer use'])
    df_features['training_encoded'] = (df['Prior cybersecurity training?'] == 'Yes').astype(int)
    df_features['multitask_encoded'] = le_multitask.fit_transform(df['Multitasking frequency'])
    
    # Numeric features
    numeric_features = [
        'On a scale of 1-5, how important is data privacy to you in your professional role?',
        'How confident are you in identifying a phishing email?',
        'How would you rate your overall digital/computer literacy?',
        'I feel stressed at work most days.',
        'I tend to follow requests that appear to come from managers.'
    ]
    
    for col in numeric_features:
        df_features[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Final feature matrix
    feature_columns = [
        'age_encoded',
        'experience_encoded',
        'training_encoded',
        'multitask_encoded',
        'On a scale of 1-5, how important is data privacy to you in your professional role?',
        'How confident are you in identifying a phishing email?',
        'How would you rate your overall digital/computer literacy?',
        'I feel stressed at work most days.',
        'I tend to follow requests that appear to come from managers.'
    ]
    
    X = df_features[feature_columns].copy()
    
    # Fill missing values with median (numeric columns only)
    for col in feature_columns:
        if X[col].dtype in ['int64', 'float64']:
            X[col] = X[col].fillna(X[col].median())
    
    # Target variable (binary: High Risk = 1, Low/Medium = 0)
    y = (df['risk_level'] == 'High').astype(int)
    
    print(f"  ✓ Feature matrix: {X.shape}")
    print(f"  ✓ Target distribution: {y.value_counts().to_dict()}")
    
    # Save encoders
    encoders = {
        'age_encoder': le_age,
        'experience_encoder': le_exp,
        'multitask_encoder': le_multitask
    }
    
    return X, y, encoders


def train_models(X, y):
    """Train multiple ML models"""
    print("\n🤖 Training ML models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Handle class imbalance with SMOTE
    print("\n  Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"  ✓ Balanced training set: {len(X_train_balanced)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        'SVM': SVC(
            kernel='rbf',
            probability=True,
            C=1.0,
            gamma='scale',
            random_state=42
        )
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train_balanced)
        
        # Cross-validation on training set
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train_balanced,
            cv=5, scoring='roc_auc'
        )
        
        # Predictions on test set
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'cv_mean_roc_auc': float(cv_scores.mean()),
            'cv_std_roc_auc': float(cv_scores.std()),
            'test_roc_auc': float(roc_auc),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        trained_models[name] = model
        
        print(f"    CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"    Test ROC-AUC: {roc_auc:.4f}")
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['test_roc_auc'])
    best_model = trained_models[best_model_name]
    
    print(f"\n  ✅ Best model: {best_model_name} (ROC-AUC: {results[best_model_name]['test_roc_auc']:.4f})")
    
    return trained_models, scaler, results, best_model_name, (X_test, y_test)


def save_models(trained_models, scaler, encoders, results, best_model_name):
    """Save trained models and artifacts"""
    print("\n💾 Saving models and artifacts...")
    
    # Save best model
    best_model_file = output_dir / 'vulnerability_predictor_synthetic.joblib'
    joblib.dump({
        'model': trained_models[best_model_name],
        'scaler': scaler,
        'encoders': encoders,
        'model_name': best_model_name
    }, best_model_file)
    print(f"  ✓ Saved best model: {best_model_file}")
    
    # Save all models
    all_models_file = output_dir / 'all_models_synthetic.joblib'
    joblib.dump({
        'models': trained_models,
        'scaler': scaler,
        'encoders': encoders
    }, all_models_file)
    print(f"  ✓ Saved all models: {all_models_file}")
    
    # Save results
    results_file = output_dir / 'model_results_synthetic.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved results: {results_file}")


def main():
    """Main training pipeline"""
    
    print("="*70)
    print("ML MODEL TRAINING - SYNTHETIC DATA")
    print("Training vulnerability prediction models on 400+ responses")
    print("="*70)
    
    # Load data
    df = load_processed_data()
    
    # Prepare features
    X, y, encoders = prepare_features(df)
    
    # Train models
    trained_models, scaler, results, best_model_name, test_data = train_models(X, y)
    
    # Save models
    save_models(trained_models, scaler, encoders, results, best_model_name)
    
    print("\n" + "="*70)
    print("✅ MODEL TRAINING COMPLETE!")
    print("="*70)
    
    print(f"\n📊 Model Performance Summary:")
    for name, result in results.items():
        marker = " ⭐" if name == best_model_name else ""
        print(f"\n  {name}{marker}:")
        print(f"    CV ROC-AUC: {result['cv_mean_roc_auc']:.4f}")
        print(f"    Test ROC-AUC: {result['test_roc_auc']:.4f}")
        print(f"    Precision: {result['classification_report']['1']['precision']:.4f}")
        print(f"    Recall: {result['classification_report']['1']['recall']:.4f}")
        print(f"    F1-Score: {result['classification_report']['1']['f1-score']:.4f}")
    
    print(f"\n🎯 Best Model: {best_model_name}")
    print(f"   ROC-AUC: {results[best_model_name]['test_roc_auc']:.4f}")
    
    print(f"\n📁 Saved Files:")
    print(f"  • models/saved/vulnerability_predictor_synthetic.joblib")
    print(f"  • models/saved/all_models_synthetic.joblib")
    print(f"  • models/saved/model_results_synthetic.json")
    
    print(f"\n🚀 Next Steps:")
    print(f"  1. Test model predictions:")
    print(f"     python src/models/test_synthetic_model.py")
    print(f"  2. View dashboard with predictions:")
    print(f"     streamlit run src/visualization/synthetic_data_dashboard.py")
    print(f"  3. Calculate SHAP values:")
    print(f"     python src/models/calculate_shap_synthetic.py")
    
    print(f"\n⚠️  Note: Trained on SYNTHETIC DATA (400+ responses)")
    print(f"   These models demonstrate the pipeline and expected performance.")
    print(f"   For IEEE publication, retrain on real participant data.")


if __name__ == "__main__":
    main()
