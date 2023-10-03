import yaml,os,sys
from income.exception import IncomeException


def read_yaml(file_path:str):
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise IncomeException(sys,e) from e