import argparse
import os
import logging
from src.utils import read_yaml, create_directories, process_posts
import random


STAGE = "prepare_data" ## <<< change stage name 

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
    
    source_data = config["source_data"]
    source_data_dir = config["source_data"] ["data_dir"]
    source_data_file = config["source_data"] ["data_file"]
    source_data_path = os.path.join(source_data_dir, source_data_file)
    
    split = params["prepare"] ["split"] #split ratio from params file
    seed = params["prepare"] ["seed"] #from params file
    tag = params["prepare"] ["tag"]

    random.seed(seed)

    artifacts = config["artifacts"]
    prepare_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["PREPARED_DATA"])
    create_directories([prepare_data_dir_path])

    train_data_path = os.path.join(prepare_data_dir_path, artifacts["TRAIN_DATA"])
    test_data_path = os.path.join(prepare_data_dir_path, artifacts["TEST_DATA"])

    encode="utf8"

    with open(source_data_path, encoding=encode) as fd_in: #actual input data that we are reading
        with open(train_data_path,"w", encoding=encode) as fd_out_train: #writing the train data
            with open(test_data_path, "w", encoding=encode) as fd_out_test: #writing the test data
                process_posts(fd_in, fd_out_test, fd_out_train, tag, split)



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