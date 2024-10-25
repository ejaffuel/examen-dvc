import pandas as pd 
import numpy as np
import pickle 
import json
from pathlib import Path

from check_structure import check_existing_file

X_test = pd.read_csv('data/preprocessed/X_test.csv')
y_test = pd.read_csv('data/preprocessed/y_test.csv')
y_test = np.ravel(y_test)

def save_dataframe(dataframe, output_filepath):
    if check_existing_file(output_filepath):
        dataframe.to_csv(output_filepath, index=False)

def main(repo_path):
    model = pickle.load(repo_path / "models/gbr_model.pkl")
    predictions = model.predict(X_test)
    score = modele.score(X_test, y_test)
    metrics = {"score": score}
    metric_path = repo_path / "metrics/scores.json"
    metric_path.write_text(json.dumps(metrics))

    save_dataframe(predictions, "predict/prediction.csv")

if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent
    main(repo_path)