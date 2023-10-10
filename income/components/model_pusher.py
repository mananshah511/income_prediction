import os,sys,shutil
from income.loggers import logging
from income.exception import IncomeException
from income.entity.config_entity import ModelPusherConfig
from income.entity.artifact_entity import ModelEvulationArtifact,ModelPusherArtifact


class ModelPusher:

    def __init__(self,model_pusher_config:ModelPusherConfig,
                 model_Evulation_artifact:ModelEvulationArtifact) -> None:
        try:
            logging.info(f"{'>>'*20}Model Pusher log started.{'<<'*20} \n\n")
            self.model_pusher_config = model_pusher_config
            self.model_evulation_artifact = model_Evulation_artifact
        except Exception as e:
            raise IncomeException(sys,e) from e
        
    def export_model_dir(self)->ModelPusherArtifact:
        try:
            logging.info(f"export model dir function started")
            transform_train_models = self.model_evulation_artifact.evulation_model_file_path
            logging.info(f"trained models are at: {transform_train_models}")
            export_dir = self.model_pusher_config.export_dir_path

            export_dir_list = []

            for cluster_number in range(len(transform_train_models)):
                train_file_name = os.path.basename(transform_train_models[cluster_number])
                export_dir_path = os.path.join(export_dir,'cluster'+str(cluster_number))
                os.makedirs(export_dir_path,exist_ok=True)
                shutil.copy(src = transform_train_models[cluster_number],dst=export_dir_path)
                export_dir_list.append(os.path.join(export_dir_path,train_file_name))
            logging.info(f"all models are copied to: {export_dir}")
            model_pusher_artifact = ModelPusherArtifact(export_dir_path=export_dir_list)
            return model_pusher_artifact
        except Exception as e:
            raise IncomeException(sys,e) from e
        
    def intiate_model_pusher(self)->ModelPusherArtifact:
        try:
            logging.info(f"intiate model pusher function started")
            return self.export_model_dir()
        except Exception as e:
            raise IncomeException(sys,e) from e

    def __del__(self):
        logging.info(f"{'>>'*20}Model Pusher log completed.{'<<'*20} \n\n")