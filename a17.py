"""
First of all, I want to be sure that all needed libraries are installed.
And if not, I need to install them.
"""
import sys
import pip
import os

packages = ['numpy', 'pandas', 'sklearn', 'xgboost', 'optuna']

for pack in packages:
    if not pack in sys.modules.keys():
        if (pack == 'xgboost') | (pack == 'optuna') :
            os.system(f'conda install -c conda-forge {pack}')
        else:
            os.system(f'pip install {pack}')


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error
import optuna


"""
Here I ignore warning messages
"""
import warnings
warnings.filterwarnings("ignore")


class A17():

  def __init__(self, trials = 20, n_jobs = -1):

      self.median_ = None
      self.enc_ = None
      self.fitted_ = None
      self.predicted_ = None
      self.log_rmse = None
      self.trials = trials
      self.jobs = n_jobs

  def __repr__(self):
      return "This is my AUTOML-Regression Model"

  def fit(self, X, y):
      """
      Here I make preprocessing operations, in order:
      1 - Remove NaN values
      2 - Label Encoding
      3 - Hyperparameters Optimization
      """
      
      # I create a dataframe where I could save the median of numeric columns
      medians = pd.DataFrame(columns = X.columns.tolist())
      medians.loc[0] = 0
      
      for col in X.columns:
          if str(X[col].dtype) == 'object':
              X[col].fillna("NONE", inplace = True)
          else:
              X[col].fillna(X[col].median(), inplace = True)
              medians.loc[0, col] = X[col].median()
              
      
      self.median_ = medians # save the median df into model attribute

      le = LabelEncoder()
      for col in X.columns:
          if str(X[col].dtype) == 'object':
              X[col] = le.fit_transform(X[col])

      self.enc_ = le # save the Encoder into model attribute

      X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.33, random_state = 17)

      def objective(trial):
          n_estimators =  trial.suggest_int('n_estimators', 10, 1000)
          max_depth = trial.suggest_int('max_depth', 1, 100)
          eta = trial.suggest_float('eta', 0.01, 0.5)
          subsample = trial.suggest_float('subsample', 0.1, 1.0)
          colsample_bytree = trial.suggest_float('colsample_bytree', 0.1, 1.0)
        
        
          regr = XGBRegressor(n_estimators = n_estimators, 
                              max_depth = max_depth,
                              eta = eta,
                              subsample = subsample,
                              colsample_bytree = colsample_bytree,
                              n_jobs = self.jobs)
        
          regr.fit(X_train, y_train)
          y_pred = regr.predict(X_val)
          self.log_rmse = np.sqrt(mean_squared_log_error(y_val, abs(y_pred))) # RITORNIAMO IL log_rmse
          return np.sqrt(mean_squared_log_error(y_val, abs(y_pred))) #RMSE_log
      
        
      #Execute optuna and set hyperparameters
      study = optuna.create_study(direction='minimize')
      study.optimize(objective, n_trials = self.trials)

      #Create an instance with tuned hyperparameters
      optimised_reg = XGBRegressor(n_estimators = study.best_params['n_estimators'],
                                  max_depth = study.best_params['max_depth'], 
                                  eta = study.best_params['eta'],
                                  subsample = study.best_params['subsample'],
                                  colsample_bytree = study.best_params['colsample_bytree'],
                                  n_jobs = self.jobs)
    
      optimised_reg.fit(X_train ,y_train)
      self.fitted_ = optimised_reg # save fitted model into attribute
    


  def predict(self, X):
      """
      I do preprocessing also on Test Set
      """
      med = self.median_ # MEDIAN
      encoding = self.enc_ # ENCODER
      opt_reg = self.fitted_ # MODEL
    
      for col in X.columns:
          if str(X[col].dtype) == 'object':
              X[col].fillna("NONE", inplace = True)
          else:
              X[col].fillna(med.loc[0, col], inplace = True) # FILL WITH MEDIAN
  
  
      """
      This Encoder can manage also unseen values, setting these as UNKNOWN's
      """
      for col in X.columns:
          if str(X[col].dtype) == 'object':
              X[col] = X[col].map(lambda s: "UNKNOWN" if s not in encoding.classes_ else s)
              encoding.classes_ = np.append(encoding.classes_, 'UNKNOWN')
              X[col] = encoding.transform(X[col])
  
  
      self.predicted_ = opt_reg.predict(X) # save predictions in model attribute
  
      return self.predicted_