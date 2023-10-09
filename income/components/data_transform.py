import os,sys,dill,csv
from income.loggers import logging
from income.exception import IncomeException
from income.entity.artifact_entity import DataIngestionArtifact,DataTransformArtifact,DataValidationArtifact
from income.entity.config_entity import DataTransformConfig
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.preprocessing import StandardScaler
from income.constant import CATEGORICAL_COLUMN_KEY,TARGET_COLUMN_KEY,COLUMN,NO_CLUSTER
from income.util.util import read_yaml
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from pathlib import Path



global categorical_columns
class trans(BaseEstimator,TransformerMixin):
    def __init__(self,data_validation_artifact:DataValidationArtifact):
        self.data_validation_artifact = data_validation_artifact
        self.categorical_columns = read_yaml(data_validation_artifact.schema_file_path)[CATEGORICAL_COLUMN_KEY]
        self.categorical_columns = list(self.categorical_columns)
        self.categorical_columns.pop(1)
        self.categorical_columns.pop(-1)

    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        X=pd.DataFrame(X,columns=COLUMN[:-1])
        X=pd.get_dummies(X,columns = self.categorical_columns,drop_first=True,dtype='int64')
        X.drop('education',axis=1,inplace=True)
        global column_trans
        column_trans = X.columns
        logging.info(f"columns after transformation are :{column_trans}")
        return X

class DataTransform:
     
    def __init__(self,data_transform_config:DataTransformConfig,
                 data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_artifact:DataValidationArtifact) -> None:
        try:
            self.data_transform_config = data_transform_config
            self.data_ingestion_artifact = data_ingestion_artifact
            logging.info(f"{'>>'*20}Data Transformation log started.{'<<'*20} ")
            self.data_validation_artifact = data_validation_artifact
            self.target_column = read_yaml(self.data_validation_artifact.schema_file_path)[TARGET_COLUMN_KEY]
        except Exception as e:
            raise IncomeException(sys,e) from e
        
    def get_preprocessing_object(self):
        try:
            logging.info(f"get preprocessing object function started")

            logging.info(f"preprocessing pipeline esamble started")
            pipeline = Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent',missing_values=' ?')),
                                       ('dummies',trans(data_validation_artifact=self.data_validation_artifact)),
                                       ('scaler',StandardScaler())])
            logging.info(f"preprocssing pipeline esamble completed")
            return pipeline
        except Exception as e:
            raise IncomeException(sys,e) from e
        
    def perform_preprocessing(self,preprocessing_obj:Pipeline,is_test_data:bool=False,dummy_df:pd.DataFrame=None):
        try:
            logging.info(f"perform preprocessig function started")
            categorical_columns = read_yaml(self.data_validation_artifact.schema_file_path)[CATEGORICAL_COLUMN_KEY]
            categorical_columns = list(categorical_columns)
            categorical_columns.pop(1)

            if is_test_data == False:

                logging.info(f"-----reading train data started-----")    
                train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
                logging.info(f"-----reading train data completed-----")

                
                target_df = train_df.iloc[:,-1]
                logging.info("dropping target column in train dataset")
                train_df.drop(self.target_column,axis=1,inplace=True)
                dummy_df = train_df

                columns = train_df.columns
                logging.info(f"column name after dropping target columns: {columns}")

                train_df = preprocessing_obj.fit_transform(train_df)
                logging.info(f"preprocessing on train data is completed")
                
                train_df = pd.DataFrame(train_df,columns=column_trans)
                target_df = pd.get_dummies(target_df.values,drop_first=True,dtype='int64')
                logging.info(f"{target_df.columns}")
                target_column = []
                target_column.append(self.target_column)
                target_df.columns = target_column
                
                train_df = pd.concat([train_df,target_df],axis=1)
                logging.info(f"combining train and target dataframes")
                

                return train_df,preprocessing_obj,dummy_df
            
            else:

                logging.info(f"-----reading test data started-----")    
                test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
                logging.info(f"-----reading test data completed-----")

                target_df = test_df.iloc[:,-1]
                logging.info("dropping target column in test dataset")
                test_df.drop(self.target_column,axis=1,inplace=True)

                columns = test_df.columns
                logging.info(f"column name after dropping target columns: {columns}")
                len_test_df = len(test_df)
            
                temp_df = pd.concat([test_df,dummy_df])
                temp_df = preprocessing_obj.transform(temp_df)
                logging.info(f"preprocessing on test data is completed")

                test_df = pd.DataFrame(temp_df,columns=column_trans)
                test_df = test_df.iloc[:len_test_df]
                target_df = pd.get_dummies(target_df.values,drop_first=True,dtype='int64')
                target_column = []
                target_column.append(self.target_column)
                target_df.columns = target_column
                test_df = pd.concat([test_df,target_df],axis=1)
                logging.info(f"combining test and target dataframes")

                return test_df

        except Exception as e:
            raise IncomeException(sys,e) from e
        
    def get_and_save_graph_cluster(self,train_df:pd.DataFrame):
        try:
            logging.info(f"get and save no of clutser graph function started")

            logging.info(f"making k-means object")
            k_means = KMeans(init='k-means++',random_state=42)

            logging.info(f"making visulaizer object and fitting train data")
            visulizer = KElbowVisualizer(k_means,k=(2,11))
            visulizer.fit((train_df.drop(self.target_column,axis=1)))

            graph_dir = self.data_transform_config.graph_save_dir
            os.makedirs(graph_dir,exist_ok=True)
            graph_file_path = os.path.join(graph_dir,'graph_clutser.png')
            visulizer.show(graph_file_path)

            logging.info(f"graph saved successfully")

        except Exception as e:
            raise IncomeException(sys,e) from e
        
    def get_and_save_silhouette_score_graph(self,train_df:pd.DataFrame):
        try:
            logging.info(f"get and save silhouetter score graph function started")
            fig, ax = plt.subplots(2, 2, figsize=(15,8))
            for no_clusters in [2,3,4,5]:
                logging.info(f"finding and saving graph for silhouette score for {no_clusters} clusters")
                kmeans = KMeans(n_clusters=no_clusters,init='k-means++',random_state=42)
                q, mod = divmod(no_clusters, 2)
                visulizer = SilhouetteVisualizer(kmeans,colors='yellowbrick',ax=ax[q-1][mod])
                visulizer.fit((train_df.drop(self.target_column,axis=1)))

                graph_dir = self.data_transform_config.graph_save_dir
                os.makedirs(graph_dir,exist_ok=True)
                graph_file_path = os.path.join(graph_dir,'cluster_'+str(no_clusters)+'_silhouette_score.png')
                visulizer.show(graph_file_path)
                logging.info(f"graph saved successfully")

        except Exception as e:
            raise IncomeException(sys,e) from e
        
    def save_data_based_clusters(self,train_df:pd.DataFrame,test_df:pd.DataFrame,n_clusters):
        try:
            logging.info(f"save data based on cluster function started")

            logging.info(f"making cluster object and fittig train data")
            k_means = KMeans(n_clusters=n_clusters,init='k-means++',random_state=42)
            k_means.fit((train_df.drop(self.target_column,axis=1)))

            logging.info(f"prediction train data's cluster")
            train_predict = k_means.predict((train_df.drop(self.target_column,axis=1)))

            transform_train_folder = self.data_transform_config.transform_train_dir
            os.makedirs(transform_train_folder,exist_ok=True)

            column_name = list(train_df.columns)
            logging.info(f"train column names are : {column_name}")

            cluster_numbers = list(np.unique(np.array(train_predict)))
            logging.info(f"cluster numbers are : {cluster_numbers}")

            logging.info(f"making csv files for train data cluster wise")

            for cluster_number in cluster_numbers:
                train_file_path = os.path.join(transform_train_folder,'train_cluster'+str(cluster_number)+'.csv')
                with Path(train_file_path).open('w',newline='') as csvfiles:
                    csvwriter = csv.writer(csvfiles)

                    csvwriter.writerow(column_name)
                    for index in range(len(train_predict)):
                        if train_predict[index] == cluster_number:
                            csvwriter.writerow(train_df.iloc[index])
            logging.info(f"csv files write for train data is completed")


            logging.info(f"prediction test data's cluster")
            test_predict = k_means.predict((test_df.drop(self.target_column,axis=1)))

            transform_test_folder = self.data_transform_config.transform_test_dir
            os.makedirs(transform_test_folder,exist_ok=True)

            column_name = list(test_df.columns)
            logging.info(f"test column names are : {column_name}")

            cluster_numbers = list(np.unique(np.array(test_predict)))
            logging.info(f"cluster numbers are : {cluster_numbers}")

            logging.info(f"making csv files for test data cluster wise")

            for cluster_number in cluster_numbers:
                test_file_path = os.path.join(transform_test_folder,'test_cluster'+str(cluster_number)+'.csv')
                with Path(test_file_path).open('w',newline='') as csvfiles:
                    csvwriter = csv.writer(csvfiles)

                    csvwriter.writerow(column_name)
                    for index in range(len(test_predict)):
                        if test_predict[index] == cluster_number:
                            csvwriter.writerow(test_df.iloc[index])
            logging.info(f"csv files write for test data is completed")

            return k_means
        except Exception as e:
            raise IncomeException(sys,e) from e
        
    def intiate_data_transform(self)->DataTransformArtifact:
        try:
            logging.info(f"intiate data transform function started")

            preprocessing_obj = self.get_preprocessing_object()
            train_df, preprocessing_obj, dummy_df = self.perform_preprocessing(preprocessing_obj=preprocessing_obj)

            logging.info(f"saving preprocessing object")
            preprocessing_dir = os.path.dirname(self.data_transform_config.preprocessed_file_path)
            os.makedirs(preprocessing_dir,exist_ok=True)
            with open(self.data_transform_config.preprocessed_file_path,'wb') as objfile:
                dill.dump(preprocessing_obj,objfile)
            logging.info(f"preprocessing object saved")

            test_df = self.perform_preprocessing(preprocessing_obj=preprocessing_obj,is_test_data=True,dummy_df=dummy_df)

            self.get_and_save_graph_cluster(train_df=train_df)
            self.get_and_save_silhouette_score_graph(train_df=train_df)

            k_means = self.save_data_based_clusters(train_df=train_df,test_df=test_df,n_clusters=NO_CLUSTER)

            logging.info(f"saving cluster object")
            cluster_dir = os.path.dirname(self.data_transform_config.cluster_model_file_path)
            os.makedirs(cluster_dir,exist_ok=True)
            with open(self.data_transform_config.cluster_model_file_path,'wb') as objfile:
                dill.dump(k_means,objfile)
            logging.info(f"cluster object saved")

            data_transform_artifact = DataTransformArtifact(transform_train_dir=self.data_transform_config.transform_train_dir,
                                                            transform_test_dir=self.data_transform_config.transform_test_dir,
                                                            is_transform=True,
                                                            message="Data Transform completed",
                                                            preprocessed_dir=self.data_transform_config.preprocessed_file_path,
                                                            cluster_model_dir=self.data_transform_config.cluster_model_file_path)
            return data_transform_artifact           

        except Exception as e:
            raise IncomeException(sys,e) from e
        
    def __del__(self):
        logging.info(f"{'>>'*20}Data Transformation log completed.{'<<'*20} \n\n")


