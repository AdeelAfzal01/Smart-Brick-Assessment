# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, OrdinalEncoder
import pickle
import os
import re
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE, SelectKBest
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from sklearn.linear_model import LinearRegression
                
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

def ApplyOneHotEncoder(df, cols, save_path="encoders", mode="train"):
    os.makedirs(save_path, exist_ok=True)

    if mode == "train":
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = ohe.fit_transform(df[cols])
        joblib.dump(ohe, f"{save_path}/OneHotEncoder.pkl")
    elif mode == "test":
        ohe = joblib.load(f"{save_path}/OneHotEncoder.pkl")
        encoded_data = ohe.transform(df[cols])

    # Convert encoded data to DataFrame
    encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(cols))
    df.drop(cols, axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, encoded_df], axis=1)
    return df

def ApplyOrdinalEncoder(df, cols, save_path="encoders", mode="train"):
    os.makedirs(save_path, exist_ok=True)

    if mode == "train":
        # Train and save label encoders
        for col in cols:
            le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df[col] = le.fit_transform(df[[col]])
            joblib.dump(le, f"{save_path}/OrdinalEncoder_{col}.pkl")
    elif mode == "test":
        # Load and apply pre-trained label encoders
        for col in cols:
            le = joblib.load(f"{save_path}/OrdinalEncoder_{col}.pkl")
            df[col] = le.transform(df[[col]])

def ScaleData(df, columns_to_scale, save_path="scaler.pkl", mode="train"):
    if mode == "train":
        scaler = StandardScaler()
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        with open(save_path, "wb") as f:
            pickle.dump(scaler, f)
            
    elif mode == "test":
        with open(save_path, "rb") as f:
            scaler = pickle.load(f)
        df[columns_to_scale] = scaler.transform(df[columns_to_scale])


## Feature Selection
def CorrelationAnalysis(df, threshold=0.9):
    corr_matrix = df.corr()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column].abs() > threshold)]
    correlated_features = df.drop(to_drop, axis=1).columns.tolist()
    return correlated_features

def FeatureImportance(df, target_column, n_features=10, save_path="feature_importance_model"):

    X = df.drop(columns=[target_column])
    y = df[target_column]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)    
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    feature_importances = feature_importances.sort_values(ascending=False)
    top_features = feature_importances.head(n_features).index.tolist()
    return top_features

def RecursiveFeatureElimination(df, target_column, n_features=10, save_path="rfe_model"):
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    model = LinearRegression()
    selector = RFE(model, n_features_to_select=n_features)
    selector.fit(X, y)
    selected_features = X.columns[selector.support_].tolist()  
    return selected_features

def UnivariateFeatureSelection(df, target_column, k=10):

    X = df.drop(columns=[target_column])
    y = df[target_column]
    selector = SelectKBest(f_regression, k=k)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    return selected_features

def CombineSelectedFeatures(correlation_features, importance_features, rfe_features, univariate_features):
    combined_features = set(correlation_features) | set(importance_features) | set(rfe_features) | set(univariate_features)
    final_selected_features = list(combined_features)
    return final_selected_features


def DropColumns(df, cols=[]):
    if cols:
        df.drop(cols, axis=1, inplace=True)
    else:
        pass
    return df
