import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils import read_yaml, save_json
import random
import math
import sklearn.metrics as metrics
import numpy as np
import joblib

STAGE = "Evaluation" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    artifacts = config["artifacts"]
    featurized_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["FEATURIZED_DATA"])
    featurized_test_data_path = os.path.join(featurized_data_dir_path, artifacts["FEATURIZED_DATA_TEST"])


    model_dir = artifacts["MODEL_DIR"]
    model_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], model_dir)
    model_name = artifacts["MODEL_NAME"]
    model_path = os.path.join(model_dir_path, model_name)

#Read the model
    model = joblib.load(model_path)
    metrix = joblib.load(featurized_test_data_path)

    labels = np.squeeze(metrix[:, 1].toarray())
    X = metrix[:, 2:] #second row onwards
   
#Prediction
    predictions_probability = model.predict_probo(X)

    pred = predictions_probability[:,1]

    logging.info(f"labels, predictions: {list(zip(labels, pred))}")
    
    #print(np.unique(predictions_probability), np.unique(labels))

    PRC_json_path = config["plots"] ["PRC"]
    ROC_json_path = config["plots"] ["ROC"]
    scores_json_path = config["matrics"] ["SCORES"]

    avg_prec = metrics.average_precision_score(labels, pred)
    roc_auc = metrics.roc_auc_score(labels, pred)
    
    logging.info(f"len of labels: {len(labels)} and predictions: {len(pred)}")
    
    scores = {
        "avg_prec": avg_prec,
        "roc_auc": roc_auc
    }

    save_json(scores_json_path, scores)

    precision, recall, prc_threshold = metrics.precision_recall_curve(labels, pred)

    nth_points = math.ceil(len(prc_threshold)/1000)
    prc_points = list(zip(precision, recall, prc_threshold))[::nth_points]
    prc_points = list(zip(precision, recall, prc_threshold))

    
    prc_data = { 
        "prc": [
            {"precision": p, "recall": r, "threshold": t}
            for p, r, t in prc_points
        ]
    }

    logging.info(f"No. of prc points: {len(prc_points)}")

    #logging.info(f"\nprecision: {precision}, \nrecall: {recall}, \nprc_thrashold: {prc_thrashold}")

    save_json(PRC_json_path, prc_data)

    fpr, tpr, roc_threshold = metrics.roc_curve(labels, pred)

    roc_data = {
        "roc": [
            {"fpr": fp, "tpr": tp, "threshold":t}
            for fp, tp, t in zip(fpr, tpr, roc_threshold)
        ]
    }

    #logging.info(f"fpr {fpr}, \ntpr {tpr}, \nroc_threshold {roc_threshold}")

    save_json(ROC_json_path, roc_data)




if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e