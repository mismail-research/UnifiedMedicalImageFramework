import numpy as np

def fuse_features(handcrafted, deep, alpha=0.5):
    """
    Fuse handcrafted and deep features with optional attention weight alpha
    """
    fused = alpha * handcrafted + (1 - alpha) * deep
    # Z-score normalization
    fused = (fused - np.mean(fused)) / np.std(fused)
    return fused
