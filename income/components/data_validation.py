import os,sys
from income.loggers import logging
from income.exception import IncomeException
from income.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from income.entity.config_entity import DataValidationConfig
from income.util.util import read_yaml
import pandas as pd
from income.constant import COLUMN_KEY,NUMERIC_COULMN_KEY,CATEGORICAL_COLUMN_KEY
from evidently.dashboard.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab

class DataValidation:

    def __init__(self,data_ingestion_artifact : DataIngestionArtifact,
                 data_validation_config : DataValidationConfig) -> None:
        try:
            logging.info(f"{'>>'*20}Data Validation log started.{'<<'*20} \n\n")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
        except Exception as e:
            raise IncomeException(sys,e) from e
        
    def get_train_test_dataframe(self):
        try:
            logging.info(f"get train and test dataframe function started")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info(f"data loaded successfully")
            return train_df,test_df
        except Exception as e:
            raise IncomeException(sys,e) from e
        
    def check_train_test_dir_exist(self):
        try:
            logging.info(f"check train test dir exist function started")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_flag = False
            test_flag = False

            if os.path.exists(train_file_path):
                logging.info(f"train dir is available")
                train_flag = True

            if os.path.exists(test_file_path):
                logging.info(f"test dir is available")
                test_flag = True

            if train_flag == False:
                logging.info(f"train dir is not available")

            if test_flag == False:
                logging.info(f"test dir is not available")

            return train_flag and test_flag
        except Exception as e:
            raise IncomeException(sys,e) from e
        
    def check_column_count_validation(self):
        try:
            logging.info(f"check column count validation function started")
            schema_file_path = self.data_validation_config.schema_file_dir
            schema_file_content = read_yaml(file_path=schema_file_path)

            train_df,test_df = self.get_train_test_dataframe()

            train_column_count = len(train_df.columns)
            test_column_count = len(test_df.columns)

            schema_column_count = len(schema_file_content[COLUMN_KEY])

            logging.info(f"train column count is : {train_column_count}")
            logging.info(f"test column count is : {test_column_count}")
            logging.info(f"schema column count is : {schema_column_count}")

            train_flag = False
            test_flag = False

            if train_column_count == schema_column_count:
                logging.info(f"column count in train data is ok")
                train_flag = True

            if test_column_count == schema_column_count:
                logging.info(f"column count in test data is ok")
                test_flag = True

            if train_flag == False:
                logging.info(f"column count in train data is not correct")

            if test_flag == False:
                logging.info(f"column count in test data is not correct")

            return train_flag and test_flag

        except Exception as e:
            raise IncomeException(sys,e) from e
        
    def check_column_name_validation(self):
        try:
            logging.info(f"check column name validation function started")
            schema_file_path = self.data_validation_config.schema_file_dir
            schema_file_content = read_yaml(file_path=schema_file_path)

            train_df,test_df = self.get_train_test_dataframe()

            train_column_name = list(train_df.columns)
            test_column_name = list(test_df.columns)

            schema_column_name = list(schema_file_content[NUMERIC_COULMN_KEY])+list(schema_file_content[CATEGORICAL_COLUMN_KEY])

            logging.info(f"train column names are : {train_column_name}")
            logging.info(f"test column names are : {test_column_name}")
            logging.info(f"schema column names are : {schema_column_name}")

            train_flag = False
            test_flag = False

            if train_column_name.sort() == schema_column_name.sort():
                logging.info(f"train column names are correct")
                train_flag = True

            if test_column_name.sort() == schema_column_name.sort():
                logging.info(f"test column names are correct")
                test_flag = True

            if train_flag == False:
                logging.info(f"column names in train data is not correct")

            if test_flag == False:
                logging.info(f"column names in test data is not correct")

            return train_flag and test_flag

        except Exception as e:
            raise IncomeException(sys,e) from e
        
    def check_column_datatype_validation(self):
        try:
            logging.info(f"check column datatype validation function started")
            logging.info(f"check column name validation function started")
            schema_file_path = self.data_validation_config.schema_file_dir
            schema_file_content = read_yaml(file_path=schema_file_path)

            train_df,test_df = self.get_train_test_dataframe()

            train_data = dict(train_df.dtypes)
            test_data = dict(test_df.dtypes)

            schema_data = schema_file_content[COLUMN_KEY]
    
            train_flag = False
            test_flag = False

            for column_name in schema_data.keys():
                if schema_data[column_name] != train_data[column_name]:
                    logging.info(f"data type for {column_name} in train data is not correct")
                    return train_flag
                if schema_data[column_name] != test_data[column_name]:
                    if test_data[column_name] == 'float64':
                        continue
                    logging.info(f"data type for {column_name} in test data is not correct")
                    return test_flag
                
            logging.info("data type for train file is correct")
            logging.info("data type for test file is correct")
            train_flag=True
            test_flag=True

            return train_flag and test_flag
        except Exception as e:
            raise IncomeException(sys,e) from e
        
    def get_and_save_datadrift_report(self):
        try:
            logging.info(f"get and save datadrift function started")
            report_file_path = self.data_validation_config.report_page_file_dir
            report_dir = os.path.dirname(report_file_path)
            os.makedirs(report_dir,exist_ok=True)

            train_df,test_df = self.get_train_test_dataframe()
            dashboard = Dashboard(tabs=[DataDriftTab()])
            dashboard.calculate(train_df,test_df)
            dashboard.save(report_file_path)

            logging.info(f"report saved successfully")
        except Exception as e:
            raise IncomeException(sys,e) from e
        
    def intiate_data_validation(self):
        try:
            logging.info(f"intiate data validation function started")
            validation4 = False
            validation1 = self.check_train_test_dir_exist()
            if validation1:
                validation2 = self.check_column_count_validation()
            if validation2:
                validation3 = self.check_column_name_validation()
            if validation3:
                validation4 = self.check_column_datatype_validation()
            self.get_and_save_datadrift_report()

            data_validation_config = DataValidationArtifact(schema_file_path=self.data_validation_config.schema_file_dir,
                                                            reprot_file_path=self.data_validation_config.report_page_file_dir,
                                                            is_validated=validation4,message="Data validation completed")
            return data_validation_config
            

        except Exception as e:
            raise IncomeException(sys,e) from e
        
    def __del__(self):
        logging.info(f"{'>>'*20}Data Validation log completed.{'<<'*20} \n\n")

