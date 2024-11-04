import pandas as pd 
import numpy as np
import pickle 
import json
from pathlib import Path
import sys
import os
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import sys
import os

import dagshub
import mlflow

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.check_structure import check_existing_file, check_existing_folder
from src.config_manager import ConfigurationManager
from src.common_utils import read_yaml

config_manager = ConfigurationManager()

data_split_config = config_manager.get_data_split_config()
data_grid_search_config = config_manager.get_data_grid_search_config()
data_training_config = config_manager.get_data_training_config()
data_evaluate_config = config_manager.get_data_evaluate_config()

X_test = pd.read_csv(data_split_config.output_folderpath 
                      + data_split_config.X_test_scaled_filename)
y_test = pd.read_csv(data_split_config.output_folderpath 
                      + data_split_config.y_test_filename)
y_test = np.ravel(y_test)

def save_dataframe(dataframe, output_filepath):

    if check_existing_folder (Path(output_filepath).parent):
        os.makedirs(output_filepath)
    
    if check_existing_file(output_filepath):
        dataframe.to_csv(output_filepath, index=False)

def main():
    with open(data_training_config.model_filepath, 'rb') as f:
        modele = pickle.load(f)

    predictions = pd.DataFrame(modele.predict(X_test), X_test.index)
    predictions.columns = ['silica_concentrate_prediction']

    score = modele.score(X_test, y_test)
    rmse = root_mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    with open(data_grid_search_config.best_params_filepath, 'rb') as f:
        best_params = pickle.load(f)


    mlflow.set_tracking_uri('https://dagshub.com/ejaffuel/examen-dvc.mlflow')
    # mlflow.set_experiment("Experience")

    dagshub.auth.add_app_token(os.environ['DAGSHUB_TOKEN'])

    dagshub.init(
        repo_owner = 'ejaffuel',
        repo_name = 'examen-dvc',
        mlflow = True
    )

    from src.config import CONFIG_FILE_PATH
    params_YAML = read_yaml(CONFIG_FILE_PATH)
    with mlflow.start_run():
        mlflow.log_param("test_size",
                          params_YAML.split_config.test_size)
        mlflow.log_param("max_depth_range",
                           params_YAML.grid_search_config.max_depth_range)
        mlflow.log_param("n_estimators_range",
                           params_YAML.grid_search_config.n_estimators_range)
        mlflow.log_params(best_params)
        mlflow.sklearn.log_model(modele, artifact_path="model")
        mlflow.sklearn.log_model(best_params, artifact_path="best_params")
        mlflow.log_metric("score", score)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_artifact("data/processed")

    metrics = {
            "score" : score,
            "rmse" : rmse,
            "mae" : mae
            }
   
    metric_path = Path(data_evaluate_config.metrics_filepath)
    metric_path.write_text(json.dumps(metrics))

    output_filepath = data_evaluate_config.predictions_filepath
    save_dataframe(predictions, output_filepath)


if __name__ == "__main__":
    main()