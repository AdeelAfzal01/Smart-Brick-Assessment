# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle
import os
import re
import joblib

## Imputation
# Fill missing values with Unknown
def FillWithUnknown(data, cols):
    if isinstance(cols, list):
        for col in cols:
            data[col] = data[col].fillna("Unknown")
        
    elif isinstance(cols, str):
        data[cols] = data[cols].fillna("Unknown")

# Sort and Ffill
def FillAfterSorting(data, cols, sorting_col=""):
    if sorting_col:
        data.sort_values(by=sorting_col, inplace=True)
        if isinstance(cols, list):
            for col in cols:
                data[col] = data[col].ffill().bfill()
        
        elif isinstance(cols, str):
            data[cols] = data[cols].ffill().bfill()


# Remove outliers using IQR
def RemoveOutliers(data, cols):
    for col in cols:
        q1, q2, q3 = data.loc[:, col].quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        lower_threshold = round(q1 - (1.5 * iqr), 2)
        upper_threshold = round(q3 + (1.5 * iqr), 2)
        median = data.loc[:, col].median()
        data.loc[ (data[col] < lower_threshold) | (data[col] > upper_threshold), col] = median

## Feature Engineering 
def ConvertRoomsToNumber(value):
    
    if 'B/R' in value:
        return int(re.findall('\d+', value)[0])  # Extract the number before "B/R"
    elif 'STUDIO' in value or 'UNKNOWN' in value:
        return 0  # Represent Studio as 0 bedrooms
    elif value in ['OFFICE', 'SHOP', 'PENTHOUSE', 'SINGLE ROOM', 'HOTEL']:
        return -1  # Assign -1 or another placeholder for non-bedroom categories
    elif 'SINGLE ROOM' in value:
        return 1
    else:
        return np.nan  # Handle unexpected values

def TransactionAgeInDays(data, col):
    current_time = pd.Timestamp.now()
    data['transaction_age_in_days'] = round((current_time - pd.to_datetime(data[col])).dt.total_seconds() / (60 * 60 * 24))

def RoomsCleaner(df, col):
    df[col] = df[col].str.upper()
    df[col] = df[col].apply(ConvertRoomsToNumber)

## One Hot Encoding
def ApplyOneHotEncoder(df, cols, save_path="encoders"):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data = ohe.fit_transform(df[cols])
    encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(df[cols].columns))
    df.drop(cols, axis=1, inplace=True)
    df = pd.concat([df, encoded_df], axis=1)
    df.reset_index(drop=True, inplace=True)
    os.makedirs(save_path, exist_ok=True)
    joblib.dump(ohe, f"{save_path}/OneHotEncoder.pkl")
    return df

## Encoding Features
def ApplyLabelEncoder(df, cols, save_path="encoders"):
    os.makedirs(save_path, exist_ok=True)

    label_encoders = {}
    for col in cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Save the label encoder
    for col, le in label_encoders.items():
        joblib.dump(le, f"{save_path}/LabelEncoder_{col}.pkl")


def ScaleData(df, columns_to_scale, save_path="scaler.pkl"):
    scaler = StandardScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    with open(save_path, "wb") as f:
        pickle.dump(scaler, f)
