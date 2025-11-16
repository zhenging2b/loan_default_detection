import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

def evaluate_outlier_model(model, X_train, X_valid, y_valid,categorical_cols, numeric_cols, boolean_cols, sparse_output, ispyod=False):
    """
    Build a pipeline for outlier detection, fit it, score validation data, and evaluate performance.

    Args:
        model: An outlier detection estimator (e.g., LocalOutlierFactor, OneClassSVM, IsolationForest).
        X_train (pd.DataFrame): Training feature data.
        X_valid (pd.DataFrame): Validation feature data.
        y_valid (array-like): Validation ground truth labels (1 for outlier, 0 for normal).
        categorical_cols (list): List of categorical column names.
        numeric_cols (list): List of numeric column names.
        boolean_cols (list): List of boolean column names.
        sparse_output (bool, optional): Whether OneHotEncoder should output sparse matrix. Default is True.

    Returns:
        ap (float): Average precision score on validation set.
        roc_auc (float): ROC-AUC score on validation set.
        anom_score (np.ndarray): Normalized anomaly scores for validation set.

    Example:
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
        ap, roc_auc, scores = evaluate_outlier_model(
            lof, X_train, X_valid, y_valid, categorical_cols, numeric_cols, boolean_cols
        )
    """
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output if not ispyod else False), categorical_cols), #PYOD dont handle spare output
            ("num", StandardScaler(with_mean=False), numeric_cols),
            ("bool", "passthrough", boolean_cols),
        ],
        sparse_threshold=1.0,
    )
    pipe = Pipeline([("prep", preprocess), ("clf", model)])
    pipe.fit(X_train)

    # Score validation
    if ispyod:
        # ECOD: higher score_samples = more abnormal
        scores_normal = pipe["clf"].decision_function(pipe["prep"].transform(X_valid))

        # Normalize to [0,1]
        min_v, max_v = np.min(scores_normal), np.max(scores_normal)
        anom_score = (scores_normal - min_v) / (max_v - min_v + 1e-12)
    else:
        scores_normal = pipe["clf"].score_samples(pipe["prep"].transform(X_valid))
        raw_anom = -scores_normal  # invert: higher = more anomalous

        # Normalize to [0,1]
        min_v, max_v = np.min(raw_anom), np.max(raw_anom)
        anom_score = (raw_anom - min_v) / (max_v - min_v + 1e-12)

    # Evaluation
    ap = average_precision_score(y_valid, anom_score)
    roc_auc = roc_auc_score(y_valid, anom_score)
    return ap, roc_auc, anom_score