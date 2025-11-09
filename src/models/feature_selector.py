"""Feature selection helpers for tabular survey data."""
from typing import List
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def simple_feature_select(X: pd.DataFrame, var_thresh: float = 0.0, corr_thresh: float = 0.95) -> List[str]:
    # Remove near-constant
    selector = VarianceThreshold(threshold=var_thresh)
    x_var = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support(indices=True)])
    # Remove highly correlated features
    corr = x_var.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [column for column in upper.columns if any(upper[column] > corr_thresh)]
    selected = [c for c in x_var.columns if c not in drop_cols]
    return selected