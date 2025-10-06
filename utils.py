import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score



def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add different temporal features such as upb month on month different
    :param df: Original Clean dataframe to add temporal features
    :return: DF with temporal features
    """
    upb_cols = [f"{i}_CurrentActualUPB" for i in range(14)]
    # Create a DataFrame of month-over-month differences
    upb_diff = pd.DataFrame({
        f"{i}_{i + 1}_UPB_Diff": ((df[upb_cols[i]] - df[upb_cols[i + 1]]) /
                                  df[upb_cols[i]]) * 100
        for i in range(13)
    })
    temporal_features = pd.concat([df, upb_diff], axis=1)
    temporal_features["UPB_std"] = temporal_features[upb_cols].std(axis=1)
    temporal_features["UPB_trend"] = temporal_features[upb_cols].iloc[:, -1] - temporal_features[upb_cols].iloc[:, 0]
    temporal_features["UPB_range"] = temporal_features[upb_cols].max(axis=1) - temporal_features[upb_cols].min(axis=1)

    return temporal_features


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