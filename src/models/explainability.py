"""Explainability utilities using SHAP for vulnerability models."""
from typing import Optional
import os
import joblib
import shap
import pandas as pd


def compute_shap(model, X: pd.DataFrame, out_dir: str = "models/trained") -> str:
    os.makedirs(out_dir, exist_ok=True)
    explainer = shap.Explainer(model.predict_proba, X)
    shap_values = explainer(X)
    out_path = os.path.join(out_dir, "shap_values.pkl")
    joblib.dump({"values": shap_values.values, "base_values": shap_values.base_values, "data_columns": X.columns.tolist()}, out_path)

    # Save mean absolute SHAP as feature importance CSV
    mean_abs = pd.Series(abs(shap_values.values).mean(axis=0), index=X.columns).sort_values(ascending=False)
    mean_abs.to_csv(os.path.join(out_dir, "feature_importance.csv"))
    return out_path