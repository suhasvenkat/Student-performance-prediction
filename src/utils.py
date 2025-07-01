import sys
import os
import numpy as np
import pandas as pd
from src.exception import Customexception
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise Customexception(e,sys)
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        # Iterate over model_name and model instance directly
        for model_name, model in models.items():

            # Get params for this model safely, default empty dict
            param_grid = params.get(model_name, {})

            # Grid search
            gs = GridSearchCV(model, param_grid, cv=3)
            gs.fit(X_train, y_train)

            # Update model with best params and refit
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate scores correctly
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store test score in report dictionary
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise Customexception(e, sys)
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise Customexception(e,sys)
