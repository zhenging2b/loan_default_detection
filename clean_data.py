import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Loan data
train_df = pd.read_csv(r'orginal_data/loans_train.csv')
valid_df = pd.read_csv(r'orginal_data/loans_valid.csv')
test_df = pd.read_csv(r'orginal_data/loans_test.csv')

train_df["source"] = "train"
valid_df["source"] = "valid"
test_df["source"] = "test"
combined_df = pd.concat([train_df, valid_df, test_df])

#Start handling numerical cols
#Credit Score, 8 rows from train to clean, 3 rows from test to clean
combined_df["CreditScore_is_outlier"] = combined_df["CreditScore"] == 9999
credit_median_train = train_df.loc[train_df["CreditScore"] != 9999, "CreditScore"].median() #only use train median to avoid data leakage
combined_df.loc[combined_df["CreditScore"] == 9999, "CreditScore"] = credit_median_train
#OriginalDTI 999 for train, 1 row
combined_df["OriginalDTI_is_outlier"] = combined_df["OriginalDTI"] == 999
DTI_median_train = train_df.loc[train_df["OriginalDTI"] != 999, "CreditScore"].median()
combined_df.loc[combined_df["OriginalDTI"] == 999, "OriginalDTI"] = DTI_median_train

#Superconforming flag to fill with 'N'
combined_df["SuperConformingFlag"].fillna("N", inplace=True)
# drop all preharp and reliefrefinanceindicator (all nan), drop PPM_Flag, InterestOnlyFlag all 'N')
combined_df.drop(columns=["PreHARP_Flag", "ReliefRefinanceIndicator", "PPM_Flag", "InterestOnlyFlag"], inplace=True)


#PropertyValMethod has 1 9, ProgramIndicator has 9, BalloonIndicator 7 = not applicable
combined_df.loc[combined_df["ProgramIndicator"] == '9', "ProgramIndicator"] = "Unknown"
combined_df["PropertyValMethod"] = combined_df["PropertyValMethod"].astype(str)
combined_df.loc[combined_df["PropertyValMethod"] == "9", "PropertyValMethod"] = "Unknown"
combined_df.loc[combined_df["BalloonIndicator"] == '7', "BalloonIndicator"] = "Unknown"

cleaned_train_df = combined_df[combined_df["source"] == "train"].drop(columns=["source", "Id"])
cleaned_valid_df = combined_df[combined_df["source"] == "valid"].drop(columns=["source", "Id"])
cleaned_test_df = combined_df[combined_df["source"] == "test"].drop(columns=["source", "index", "target"])

def replace_ltv_with_estimate(row, upb_col, ltv_col):
    orig_ltv = row['OriginalLTV']
    orig_upb = row['OriginalUPB']
    
    ltv_values = row[ltv_col].astype(float).values
    upb_values = row[upb_col].astype(float).values
    mask_999 = (ltv_values == 999)

    with np.errstate(divide='ignore', invalid='ignore'):
        computed = orig_ltv * (upb_values / orig_upb)
    
    ltv_values[mask_999] = computed[mask_999]
    return ltv_values

def upd_est_LTV(df: pd.DataFrame):
    ltv_col = []
    upb_col = []
    
    for c in df.columns:
        if 'EstimatedLTV' in c:
            ltv_col.append(c)
        
        if 'CurrentActualUPB' in c:
            upb_col.append(c)

    df['EstimatedLTV_all_MissFlag'] = (df[ltv_col] == 999).all(axis=1).astype(int)
    df[ltv_col] = df.apply(replace_ltv_with_estimate, upb_col = upb_col, ltv_col = ltv_col, axis=1, result_type='expand')

    return df

cleaned_train_df = upd_est_LTV(cleaned_train_df)
cleaned_valid_df = upd_est_LTV(cleaned_valid_df)
cleaned_test_df = upd_est_LTV(cleaned_test_df)

cleaned_train_df.to_csv('clean_data/cleaned_loans_train.csv', index=False)
cleaned_valid_df.to_csv('clean_data/cleaned_loans_valid.csv', index=False)
cleaned_test_df.to_csv('clean_data/cleaned_loans_test.csv', index=False)