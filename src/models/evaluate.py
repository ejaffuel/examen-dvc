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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.check_structure import check_existing_file, check_existing_folder
from src.config_manager import ConfigurationManager

config_manager = ConfigurationManager()

data_split_config = config_manager.get_data_split_config()
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