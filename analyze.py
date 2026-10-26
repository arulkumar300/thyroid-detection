import joblib
import numpy as np

try:
    m = joblib.load('thyroid_model__3_.pkl')
    # If it's a StackingClassifier
    estimators = getattr(m, 'estimators_', [m])
    rf = estimators[0]
    if isinstance(rf, tuple):
        rf = rf[1]

    thresholds = [tree.tree_.threshold for tree in rf.estimators_]
    feats = [tree.tree_.feature for tree in rf.estimators_]
    all_feats = np.concatenate(feats)
    all_thresh = np.concatenate(thresholds)
    valid = all_feats >= 0
    unique_feats = np.unique(all_feats[valid])

    for f in unique_feats:
        f_thresh = all_thresh[(all_feats == f) & valid]
        print(f"Feature {f}: min_thresh={np.min(f_thresh):.2f}, max_thresh={np.max(f_thresh):.2f}")
except Exception as e:
    import traceback
    traceback.print_exc()
