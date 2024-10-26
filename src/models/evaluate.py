import pandas as pd 
import numpy as np
import pickle 
import json
from pathlib import Path
import sys
import os
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.check_structure import check_existing_file, check_existing_folder

X_test = pd.read_csv('data/processed/X_test_scaled.csv')
y_test = pd.read_csv('data/processed/y_test.csv')
y_test = np.ravel(y_test)

def save_dataframe(dataframe, output_folderpath, output_filepath):
    print(output_folderpath)
    print(output_filepath)

    if check_existing_folder (output_folderpath):
        os.makedirs(output_folderpath)
    
    if check_existing_file(output_filepath):
        dataframe.to_csv(output_filepath, index=False)

def main(repo_path):
    with open(repo_path / 'models/gbr_model.pkl', 'rb') as f:
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
    
    metric_path = repo_path / "metrics/scores.json"
    metric_path.write_text(json.dumps(metrics))

    output_folderpath = repo_path / "data/predict"
    output_filepath = output_folderpath / "prediction.csv"
    save_dataframe(predictions, output_folderpath, output_filepath)

if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent
    main(repo_path)