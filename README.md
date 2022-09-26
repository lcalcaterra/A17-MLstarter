# A17-MLstarter
### A17 is a simple AUTO Machine Learning Model


The idea behind this work is to create a Machine Learning model that contains some facilitations as:
- Managing NaN values
- Encoding of variables
- Hyperparameters Optimization

In this model NaN values are labeled as "NONE" in non-numerics columns and they are filled with median in numeric columns.
I performed encoding with sklearn LabelEncoder as the models (Regression version and Classification version) are tree based (XGBOOST, https://xgboost.readthedocs.io/en/stable/python/python_intro.html) and for hyperparameter optimization I used Optuna (https://optuna.org/).


### Model takes these parameters:
1. ***task***: 
    
    The task parameter is used to choose between Regression and Classification model. 
    The accepted values are *regression* (default) and *classification*

2. ***objective***: 

    I used the objective parameter to add the possibility to choose a binary or multi-labels classification. 
    As default we have a binary classification task, to perform a multi-labels classification we have to change this parameter. 
    The accepted values are *binary:logistic* (for binary classes) and *multi:softmax* (for multi-labels)

3. ***trials***:

    This parameter has used for choose the number of trials that the Optuna algorithm behind the model should perform.
    Trials is setted to 20 as default value, but you could choose an integer number of trials. 
    Obviously, much bigger as the number and much more time the model requires.

4. ***n_jobs***:

    The last parameter is the number of cores that the model can use. 
    I setted it to -1 (all cores), but you could choose a value between 1 and the max of your cores.


This is what I usually do in my daily work, you could create a different model with your pipeline. I created only a base version and in the next weeks (or months) I hope to update this project with more features.

## Use it on your script or Notebook running this command
Use pip to intall it like below. 
```bash
pip install git+https://github.com/lcalcaterra/A17-MLstarter
```
And import the model from in a notebook/script:
```
from A17.a17 import A17

reg = A17()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

### I hope you enjoyed it!
