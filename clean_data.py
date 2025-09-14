import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Loan data
train_df = pd.read_csv(r'loans_train.csv')
valid_df = pd.read_csv(r'loans_valid.csv')

#Credit Score, 8 rows from train to clean


#OriginalDTI 999 for train, 1 row

#Superconforming flag to fill with 'N'

# drop all preharp and reliefrefinanceindicator

#PropertyValMethod has 1 9