from flask import render_template,Flask,request
import os,sys,json
from flask_cors import CORS,cross_origin
from income.pipeline.pipeline import Pipeline
from income.loggers import logging
from income.exception import IncomeException
from income.entity.artifact_entity import FinalArtifact
from income.util.util import load_object
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def homepage():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    data = [x for x in request.form.values()]
    data[0] = int(data[0])
    data[1] = str(data[1])
    data[2] = float(data[2])
    data[3] = str(data[3])
    data[4] = int(data[4])
    data[5] = str(data[5])
    data[6] = str(data[6])
    data[7] = str(data[7])
    data[8] = str(data[8])
    data[9] = str(data[9])
    data[10] = float(data[10])
    data[11] = float(data[11])
    data[12] = float(data[12])
    data[13] = str(data[13])

    if not os.path.exists('data.json'):
        return render_template('index.html',output_text = "No model is trained, please start training")
    
    with open('data.json', 'r') as json_file:
        dict_data = json.loads(json_file.read())

    final_artifact = FinalArtifact(**dict_data)
    logging.info(f"final artifact : {final_artifact}")

    train_df = pd.read_csv(final_artifact.ingested_train_data)
    train_df = train_df.iloc[:,:-1]
    df = pd.DataFrame(np.array(data)).T
    df.columns = train_df.columns
    df = pd.concat([df,train_df])

    preprocessed_obj = load_object(file_path = final_artifact.preprocessed_model_path)
    df = preprocessed_obj.transform(df)
    
    df = (np.array(df[0])).reshape(1,-1)
    
    cluster_object = load_object(file_path = final_artifact.cluster_model_path)
    cluster_number = int(cluster_object.predict(df))

    model_object = load_object(file_path = final_artifact.export_dir_path[cluster_number] )
    output = int(model_object.predict(df))

    if output == 1:
        return render_template('index.html',output_text = "Given person's income is greater than 50000$ per year")
    else:
        return render_template('index.html',output_text = "Given person's income is less than 50000$ per year")

@app.route('/train',methods=['POST'])
@cross_origin()
def train():
    try:
        pipeline = Pipeline()
        pipeline.run_pipeline()
        return render_template('index.html',prediction_text = "Model training completed")
    except Exception as e:
        raise IncomeException(sys,e) from e

if __name__ == "__main__":
    app.run()