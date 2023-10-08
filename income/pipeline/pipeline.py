import os,sys
from income.loggers import logging
from income.exception import IncomeException
from income.entity.artifact_entity import DataIngestionArtifact
from income.config.configuration import Configuration
from income.constant import *
from income.components.data_ingestion import DataIngestion

class Pipeline:
    
    def __init__(self,config:Configuration=Configuration()) -> None:
        try:
            self.config = config
        except Exception as e:
            raise IncomeException(sys,e) from e
        
    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.intiate_data_ingestion()
        except Exception as e:
            raise IncomeException(sys,e) from e
        
    def run_pipeline(self):
        try:
            data_ingestion = self.start_data_ingestion()
        except Exception as e:
            raise IncomeException(sys,e) from e