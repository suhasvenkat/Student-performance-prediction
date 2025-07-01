import sys
import os
from src.exception import Customexception
import dill 
from dataclasses import dataclass
from src.logger import logging

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Decision Tree":DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "XGBClassifier":XGBRegressor(),
                "CatBoost classifier":CatBoostRegressor(),
                "AdaBoost Regressor":AdaBoostRegressor()}
            params={
                "Decision Tree":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson']
                },
                "Random Forest":{
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.8,0.85,0.9],
                    'n_estimators':[8,16,32,64,128,256],
                },
                "Linear Regression":{},
                "K-Neighbors Regressor":{
                    'n_neighbors':[3,5,7,9,11],
                },
                "XGBClassifier":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,64,128,256],
                },
                "CatBoost classifier":{
                    'depth':[6,8,10,12],
                    'learning_rate':[.1,.01,.05,.001],
                    'iterations':[20,50,100]
                },
                "AdaBoost Regressor":{
                    'n_estimators':[8,16,32,64,128,256],
                    'learning_rate':[.1,.01,.05,.001]
                }

            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)
            
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise Customexception("No best model found")
            logging.info(f"Best model found on both training and testing dataset: {best_model_name} with r2 score:{best_model_score}")
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square
        except Exception as e:
            raise Customexception(e,sys)