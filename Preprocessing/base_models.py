# base_models.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import joblib
import os

# Function to evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the model on train and test data, and return evaluation metrics.
    """
    # Training the model
    model.fit(X_train, y_train)
    
    # Predicting on train and test data
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Evaluating model
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
    print(f"Train R^2: {train_r2:.4f}, Test R^2: {test_r2:.4f}")
    
    return train_rmse, test_rmse, train_r2, test_r2

# XGBoost model
def xgboost_model(X_train, y_train, X_test, y_test, param_space=None):
    """
    Train and evaluate the XGBoost model with or without hyperparameter optimization.
    """
    # Initialize XGBoost Regressor
    model = xgb.XGBRegressor(objective="reg:squarederror")
    
    # Bayesian Optimization for Hyperparameters
    if param_space:
        opt = BayesSearchCV(model, param_space, n_iter=50, cv=3, verbose=0, n_jobs=-1)
        opt.fit(X_train, y_train)
        model = opt.best_estimator_
        print(f"Best parameters for XGBoost: {opt.best_params_}")
    
    # Evaluate model performance
    # evaluate_model(model, X_train, y_train, X_test, y_test)
    
    return model

# Random Forest model
def random_forest_model(X_train, y_train, X_test, y_test, param_space=None):
    """
    Train and evaluate the Random Forest model with or without hyperparameter optimization.
    """
    model = RandomForestRegressor(random_state=42)
    
    # Bayesian Optimization for Hyperparameters
    if param_space:
        opt = BayesSearchCV(model, param_space, n_iter=50, cv=3, verbose=0, n_jobs=-1)
        opt.fit(X_train, y_train)
        model = opt.best_estimator_
        print(f"Best parameters for Random Forest: {opt.best_params_}")
    
    # Evaluate model performance
    # evaluate_model(model, X_train, y_train, X_test, y_test)
    
    return model

# Support Vector Regression (SVR) model
def svr_model(X_train, y_train, X_test, y_test, param_space=None):
    """
    Train and evaluate the Support Vector Regression model with or without hyperparameter optimization.
    """
    model = SVR(kernel='rbf')
    
    # Bayesian Optimization for Hyperparameters
    if param_space:
        opt = BayesSearchCV(model, param_space, n_iter=50, cv=3, verbose=0, n_jobs=-1)
        opt.fit(X_train, y_train)
        model = opt.best_estimator_
        print(f"Best parameters for SVR: {opt.best_params_}")
    
    # Evaluate model performance
    evaluate_model(model, X_train, y_train, X_test, y_test)
    
    return model

# Hyperparameter Search Spaces for Bayesian Optimization
def get_param_spaces():
    """
    Define the parameter search spaces for each model.
    """
    xgboost_params = {
        'learning_rate': Real(0.01, 0.2, prior='uniform'),
        'max_depth': Integer(3, 10),
        'n_estimators': Integer(50, 300),
        'subsample': Real(0.5, 1.0, prior='uniform'),
        'colsample_bytree': Real(0.5, 1.0, prior='uniform'),
        'gamma': Real(0, 0.5, prior='uniform')
    }
    
    rf_params = {
        'n_estimators': Integer(50, 300),
        'max_depth': Integer(5, 20),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 4),
        'bootstrap': [True, False]
    }
    
    svr_params = {
        'C': Real(0.1, 1000, prior='uniform'),
        'epsilon': Real(0.01, 0.1, prior='uniform'),
        'kernel': ['linear', 'poly', 'rbf']
    }
    
    return xgboost_params, rf_params, svr_params

# Train and evaluate models with hyperparameter optimization
def train_and_evaluate_models(X, y):
    """
    Split the data, train models with hyperparameter tuning, and evaluate their performance.
    """
    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Get parameter spaces for each model
    xgboost_params, rf_params, svr_params = get_param_spaces()
    
    # Train and evaluate XGBoost model
    print("Training and Evaluating XGBoost Model:")
    xgboost_model(X_train, y_train, X_test, y_test, param_space=xgboost_params)
    
    # Train and evaluate Random Forest model
    print("Training and Evaluating Random Forest Model:")
    random_forest_model(X_train, y_train, X_test, y_test, param_space=rf_params)
    
    # Train and evaluate SVR model
    # print("Training and Evaluating Support Vector Regression Model:")
    # svr_model(X_train, y_train, X_test, y_test, param_space=svr_params)

