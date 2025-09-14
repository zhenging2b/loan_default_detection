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

cleaned_train_df.to_csv('clean_data/cleaned_loans_train.csv', index=False)
cleaned_valid_df.to_csv('clean_data/cleaned_loans_valid.csv', index=False)
cleaned_test_df.to_csv('clean_data/cleaned_loans_test.csv', index=False)