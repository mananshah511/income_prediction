import os,sys
from income.exception import IncomeException
from income.loggers import logging
from income.entity.artifact_entity import DataIngestionArtifact
from income.entity.config_entity import DataIngestionConfig
import pandas as pd
import numpy as np
from six.moves import urllib
from zipfile import ZipFile
from income.constant import DATABASE_NAME,COLUMN

class DataIngestion:

    def __init__(self,data_ingestion_config : DataIngestionConfig) -> None:
        try:
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise IncomeException(sys,e) from e
        
    def download_income_data(self)->str:
        try:
            logging.info(f"download income data function started")
            zip_data_dir = self.data_ingestion_config.zip_data_dir
            os.makedirs(zip_data_dir,exist_ok=True)

            data_downlaod_url = self.data_ingestion_config.dataset_download_url
            logging.info(f"downloading data from {data_downlaod_url} in the {zip_data_dir} folder")

            file_name = os.path.basename(data_downlaod_url)
            file_name = file_name.replace('+','_')
            zip_file_name = os.path.join(zip_data_dir,file_name)

            logging.info("-----data download started-----")
            urllib.request.urlretrieve(data_downlaod_url,zip_file_name)
            logging.info("-----data download completed-----")
            return zip_file_name
        except Exception as e:
            raise IncomeException(sys,e) from e
        
    def get_extracted_data(self,zip_file_path:str):
        try:
            logging.info(f"get extracted data function started")
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            logging.info(f"extracting data {zip_file_path} into {raw_data_dir} folder")

            os.makedirs(raw_data_dir,exist_ok=True)
            with ZipFile(zip_file_path,'r') as zip:
                zip.extractall(raw_data_dir)
            logging.info(f"data extraction completed")
        except Exception as e:
            raise IncomeException(sys,e) from e
        
    def get_train_test_split_data(self):
        try:
            logging.info(f"get train test split data function started")
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            logging.info(f"raw data dir is : {raw_data_dir}")

            raw_train_file_path = os.path.join(raw_data_dir,DATABASE_NAME+'.data')
            raw_test_file_path = os.path.join(raw_data_dir,DATABASE_NAME+'.test')

            logging.info(f"-----reading train data started-----")
            train_df = pd.read_csv(raw_train_file_path,names=COLUMN)
            logging.info(f"-----reading train data completed-----")

            logging.info(f"-----reading train data started-----")
            test_df = pd.read_csv(raw_test_file_path,names=COLUMN)
            test_df.drop(index=0,inplace=True)
            logging.info(f"-----reading train data completed-----")

            ingested_train_dir = self.data_ingestion_config.ingested_train_dir
            ingested_test_dir = self.data_ingestion_config.ingested_test_dir
            os.makedirs(ingested_train_dir,exist_ok=True)
            os.makedirs(ingested_test_dir,exist_ok=True)

            logging.info(f"ingested train dir is : {ingested_train_dir}")
            logging.info(f"ingested test dir is : {ingested_test_dir}")

            ingested_train_file_path = os.path.join(ingested_train_dir,DATABASE_NAME+'.csv')
            ingested_test_file_path = os.path.join(ingested_test_dir,DATABASE_NAME+'.csv')

            logging.info(f"saving train data as csv")
            train_df.to_csv(ingested_train_file_path)
            logging.info(f"train data saved successfully")
            logging.info(f"saving test data as csv")
            test_df.to_csv(ingested_test_file_path)
            logging.info(f"test data saved successfully")

            data_ingestion_artifact = DataIngestionArtifact(is_ingested=True,
                                                            message="data ingestion completed",
                                                            train_file_path=ingested_train_dir,
                                                            test_file_path=ingested_test_dir)
            return data_ingestion_artifact
        except Exception as e:
            raise IncomeException(sys,e) from e
        
    def intiate_data_ingestion(self):
        try:
            logging.info(f"intiate data ingestion function started")
            zip_file_path = self.download_income_data()
            self.get_extracted_data(zip_file_path=zip_file_path)
            return self.get_train_test_split_data()
        except Exception as e:
            raise IncomeException(sys,e) from e
        
    def __del__(self):
        logging.info(f"{'>>'*20}Data Ingestion log completed.{'<<'*20} \n\n")