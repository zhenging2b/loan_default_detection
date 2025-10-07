import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LinearRegression



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

def summary_stats_user(df: pd.DataFrame)-> pd.DataFrame:
    """
    Compute summary features for loan repayment behavior.
    
    Assumptions:
    - OriginalUPB is the original loan balance
    - Columns like '0_CurrentActualUPB', '1_CurrentActualUPB', ... exist
    - Lateness needs to be derived (example: on-time if scheduled balance decrease > 0)
    """

    # Extract monthly UPB columns
    upb_cols = [c for c in df.columns if "_CurrentActualUPB" in c]
    upb_cols_sorted = sorted(upb_cols, key=lambda x: int(x.split("_")[0]))
    
    # Calculate repayment ratio per month = 1 - (UPB_t / OriginalUPB)
    repayment_ratios = df[upb_cols_sorted].apply(lambda row: 1 - row / row.name, axis=1)

    # But row.name is index, fix: we'll divide by OriginalUPB
    repayment_ratios = df[upb_cols_sorted].div(df["OriginalUPB"], axis=0)
    repayment_ratios = 1 - repayment_ratios

    # 1. Average repayment ratio
    df["avg_repayment_ratio"] = repayment_ratios.mean(axis=1)

    # 2. Std deviation of repayment ratio
    df["std_repayment_ratio"] = repayment_ratios.std(axis=1)

    # 3. % months late (example heuristic: repayment ratio in month <= previous month → late/no progress)
    is_late = repayment_ratios.diff(axis=1) <= 0
    df["pct_months_late"] = is_late.sum(axis=1) / len(upb_cols_sorted)

    # 4. Trend slope (fit linear regression of repayment ratio over months)
    months = np.arange(len(upb_cols_sorted)).reshape(-1, 1)
    slopes = []
    for i in range(len(df)):
        y = repayment_ratios.iloc[i].values
        model = LinearRegression().fit(months, y)
        slopes.append(model.coef_[0])
    df["repayment_trend_slope"] = slopes

    return df

def risk_assessment(df: pd.DataFrame)-> pd.DataFrame:
    # =============================================================================
    # 1. RISK ASSESSMENT FEATURES
    # =============================================================================

    # Debt Service Ratio - actual monthly debt burden considering interest rates
    df['DebtServiceRatio'] = df['OriginalDTI'] * df['OriginalInterestRate'] / 100

    # LTV-DTI Interaction - combined leverage and debt risk
    df['LTV_DTI_Interaction'] = df['OriginalLTV'] * df['OriginalDTI']

    # Credit quality relative to leverage and debt
    df['CreditScore_LTV_Ratio'] = df['CreditScore'] / df['OriginalLTV']
    df['CreditScore_DTI_Ratio'] = df['CreditScore'] / (df['OriginalDTI'] + 1e-6)

    # Risk categories
    df['HighRiskCredit'] = (df['CreditScore'] < 620).astype(int)
    df['HighLTV'] = (df['OriginalLTV'] > 80).astype(int)
    df['HighDTI'] = (df['OriginalDTI'] > 43).astype(int)
    df['HighInterestRate'] = (df['OriginalInterestRate'] > df['OriginalInterestRate'].quantile(0.75)).astype(int)

    # Combined risk score
    df['CompositeRiskScore'] = (df['HighRiskCredit'] + 
                                        df['HighLTV'] + 
                                        df['HighDTI'] + 
                                        df['HighInterestRate'])

    df = summary_stats_user(df)
    return df


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