# A17-MLstarter
### A17 is a simple AUTO Machine Learning Model


The idea behind this work is to create a Machine Learning model that contains some facilitations as:
- Managing NaN values
- Encoding of variables
- Hyperparameters Optimization

In this model NaN values are labeled as "NONE" in non-numerics columns and are filled with median in numeric columns.
I performed encoding with sklearn LabelEncoder as the models (Regression version and Classification version) are tree based (XGBOOST, https://xgboost.readthedocs.io/en/stable/python/python_intro.html) and for hyperparameter optimization I used Optuna (https://optuna.org/).
As I said this model is based on XGBOOST Regressor and Classifier and want to be only an exercise for me, and an example for you!


### Model takes these parameters:
- task:

    The task parameter is used to choose between Regression and Classification model. 
    The accepted values are:
        - "regression" (default)
        - "classification"

- objective:

    I used the objective parameter to add the possibility to choose a binary or multi-labels classification.
    As default we have a binary classification task, to perform a multi-labels classification we have to change this parameter.
    The accepted values are:
        - "binary:logistic" (for binary classes)
        - "multi:softmax" (for multi-labels)

- trials:

    This parameter has used for choose the number of trials that the Optuna algorithm behind the model should perform.
    trials is setted to 20 as default value, bu you could choose an integer number of trials. 
    Obviously, much bigger as the number and much more time the model requires.

- n_jobs:

    The last parameter is the number of cores that the model can use. 
    I setted it to -1 (all cores), but you could choose a value between 1 and the max of your cores.


I created only a base version and in the next weeks I hope to update this project with more features.

### Use it and have FUN!
