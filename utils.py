import numpy as np
from sklearn.model_selection import StratifiedKFold

def stratified_split(X, y, n_splits=5):
    """
    Return train/test indices for 5-fold stratified cross-validation
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return skf.split(X, y)
