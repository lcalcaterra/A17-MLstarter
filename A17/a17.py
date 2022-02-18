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
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_log_error
import optuna


"""
Here I ignore warning messages
"""
import warnings
warnings.filterwarnings("ignore")

"""
Find the number of cpu cores
"""
import multiprocessing
cpus = multiprocessing.cpu_count()

l = list(range(1, cpus + 1))
l.append(-1)

"""
Dictionary of permitted values
"""
param_dict = {
  "task": ["regression", "classification"],
  "objective": ["binary:logistic", "multi:softmax"],
  "cpus": l
}


class A17():

  def __init__(self, task = "regression", objective = "binary:logistic", trials = 20, n_jobs = -1):

      self.median_ = None
      self.enc_ = None
      self.fitted_ = None
      self.predicted_ = None
      self.log_rmse = None
      self.task = task
      self.objective = objective
      self.trials = trials
      self.jobs = n_jobs

      try:
          error = ""
          if self.jobs not in param_dict['cpus']:
              error = "Error: n_jobs parameter should be a number between 1 and " + str(cpus) + " or -1 (all cores)!"

          if self.objective not in param_dict['objective']:
              error = "Error: objective parameter should be 'binary:logistic' or 'multi:softmax'!"
        
          if self.task not in param_dict['task']:
              error = "Error: task parameter should be 'regression' or 'classification'!"

          raise ValueError(error)
      except ValueError as e:
          print(e)

          
  def __repr__(self):
      return "This is my AUTOML Model. The model can perform both Regression and Classification tasks."


  def fit(self, X, y):
      """
      Here I make preprocessing operations, in order:
      1 - Remove NaN values: NaN values are substituted with string "NONE" for non-numeric values and with the median for numerics values.

      2 - Label Encoding

      3 - Hyperparameters Optimization: optimization performed with Optuna (https://optuna.org/)
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

      if self.task == "regression":

          def objective(trial):
              n_estimators =  trial.suggest_int('n_estimators', 10, 1000)
              max_depth = trial.suggest_int('max_depth', 1, 100)
              eta = trial.suggest_float('eta', 0.001, 0.5)
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
    
      elif (self.task == "classification") & (self.objective == "binary:logistic"):

          def objective(trial):
              n_estimators =  trial.suggest_int('n_estimators', 10, 1000)
              max_depth = trial.suggest_int('max_depth', 1, 100)
              learning_rate  = trial.suggest_float('learning_rate ', 0.001, 0.5)
              subsample = trial.suggest_float('subsample', 0.1, 1.0)
              colsample_bytree = trial.suggest_float('colsample_bytree', 0.1, 1.0)
              
              
              regr = XGBClassifier(n_estimators = n_estimators, 
                                  objective = self.objective,
                                  max_depth = max_depth,
                                  learning_rate  = learning_rate ,
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
          optimised_reg = XGBClassifier(n_estimators = study.best_params['n_estimators'],
                                      objective = self.objective,
                                      max_depth = study.best_params['max_depth'], 
                                      learning_rate  = study.best_params['learning_rate '],
                                      subsample = study.best_params['subsample'],
                                      colsample_bytree = study.best_params['colsample_bytree'],
                                      n_jobs = self.jobs)
          
          optimised_reg.fit(X_train ,y_train)
          self.fitted_ = optimised_reg # save fitted model into attribute


      elif (self.task == "classification") & (self.objective == "multi:softmax"):

          def objective(trial):
              n_estimators =  trial.suggest_int('n_estimators', 10, 1000)
              max_depth = trial.suggest_int('max_depth', 1, 100)
              learning_rate  = trial.suggest_float('learning_rate ', 0.001, 0.5)
              subsample = trial.suggest_float('subsample', 0.1, 1.0)
              colsample_bytree = trial.suggest_float('colsample_bytree', 0.1, 1.0)
              
              
              regr = XGBClassifier(n_estimators = n_estimators, 
                                  objective = self.objective,
                                  max_depth = max_depth,
                                  learning_rate  = learning_rate ,
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
          optimised_reg = XGBClassifier(n_estimators = study.best_params['n_estimators'],
                                      objective = self.objective,
                                      max_depth = study.best_params['max_depth'], 
                                      learning_rate  = study.best_params['learning_rate '],
                                      subsample = study.best_params['subsample'],
                                      colsample_bytree = study.best_params['colsample_bytree'],
                                      n_jobs = self.jobs)
          
          optimised_reg.fit(X_train ,y_train)
          self.fitted_ = optimised_reg # save fitted model into attribute
      
      else:
          if (self.task != "regression") | (self.task != "classification"):
              print("Error : task parameter should be 'regression' or 'classification'!")
          elif (self.objective != "binary:logistic") | (self.objective != "multi:softmax"):
              print("Error: objective parameter should be 'binary:logistic' or 'multi:softmax'!")
          else:
              print("Check the Error below!")



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
