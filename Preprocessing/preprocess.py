import sys
sys.path.append('D:\Work\Smart Brick Assessment Update\Smart-Brick-Assessment')

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
import joblib
import os
import pickle
import re
from .utils import FillWithUnknown, FillAfterSorting, RemoveOutliers, \
                  ConvertRoomsToNumber, TransactionAgeInDays, RoomsCleaner, \
                  ApplyOneHotEncoder, ScaleData, DropColumns, ApplyOrdinalEncoder

from config import FILLAFTERSORTING_COLUMNS, FILLWITHUNKNOWN_COLUMNS,\
      SORTING_COLUMN, ONEHOTENCODER_COLUMNS, LABELENCODER_COLUMNS, ROOMSEN_COLUMN,\
      TRANSACTIONDATETIME_COLUMN, REMOVEOUTLIER_COLUMNS, SCALINGFEATURES_COLUMNS,\
      TARGET_COLUMN

def Preprocess(data, mode="train"):
    # Deep Copy
    df = data.copy()
    
    # Filtering Columns
    df = df.loc[:, ONEHOTENCODER_COLUMNS + LABELENCODER_COLUMNS + [ROOMSEN_COLUMN] + [TRANSACTIONDATETIME_COLUMN] + [TARGET_COLUMN]]
    
    # Imputation
    FillAfterSorting(df, FILLAFTERSORTING_COLUMNS, SORTING_COLUMN)
    FillWithUnknown(df, FILLWITHUNKNOWN_COLUMNS)

    # Feature Engineering
    RoomsCleaner(df, ROOMSEN_COLUMN)
    TransactionAgeInDays(df, TRANSACTIONDATETIME_COLUMN)

    # Label Encoding
    ApplyOrdinalEncoder(df, LABELENCODER_COLUMNS, mode=mode)

    # One Hot Encoding
    encoded_data = ApplyOneHotEncoder(df, ONEHOTENCODER_COLUMNS, mode=mode)

    # Remove Outliers (only for training)
    if mode == "train":
        RemoveOutliers(encoded_data, REMOVEOUTLIER_COLUMNS)

    # Scale Data
    ScaleData(encoded_data, SCALINGFEATURES_COLUMNS, mode=mode)

    # Drop Columns
    final_df = DropColumns(encoded_data, [TRANSACTIONDATETIME_COLUMN])

    return final_df