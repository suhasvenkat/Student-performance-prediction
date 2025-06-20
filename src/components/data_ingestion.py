import os
import sys
from src.exception import Customexception
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import Datatransformationconfig
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact',"train.csv")
    test_data_path: str = os.path.join('artifact',"test.csv")
    raw_data_path: str = os.path.join('artifact',"raw.csv")
    
class Dataingestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion component")
        try:
            df=pd.read_csv('/Users/suhasvenkat/Projects/Student_performance_predictor/notebook/data/students.csv')
            logging.info("read the datset as dataframe")
            os.makedirs(os.path.dirname (self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info("train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=123)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of the data is comoleted")
            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)
        except Exception as e:
            raise Customexception(e,sys)
        
if __name__=="__main__":
    obj=Dataingestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))



