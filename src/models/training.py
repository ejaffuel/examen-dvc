
import pandas as pd 
from sklearn import ensemble
import pickle
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config_manager import ConfigurationManager
from src.common_utils import read_yaml

import dagshub
import mlflow
dagshub.init(repo_owner='ejaffuel', repo_name='examen-dvc', mlflow=True)

config_manager = ConfigurationManager()

data_split_config = config_manager.get_data_split_config()
data_training_config = config_manager.get_data_training_config()
data_grid_search_config = config_manager.get_data_grid_search_config()

X_train = pd.read_csv(data_split_config.output_folderpath 
                      + data_split_config.X_train_scaled_filename)
y_train = pd.read_csv(data_split_config.output_folderpath 
                      + data_split_config.y_train_filename)
y_train = np.ravel(y_train)

with open(data_grid_search_config.best_params_filepath, 'rb') as f:
    best_params = pickle.load(f)

from sklearn.ensemble import RandomForestRegressor
modele = RandomForestRegressor (
    max_depth =  best_params['max_depth'], 
    n_estimators = best_params['n_estimators']
    )

# from sklearn.ensemble import GradientBoostingRegressor
# modele = GradientBoostingRegressor (best_params)

# from sklearn.linear_model import LinearRegression
# modele = LinearRegression()

# from sklearn.linear_model import ElasticNetCV
# modele = ElasticNetCV(
#     cv = 8, # cross validation with 8-folds (Voir 2.7.9 pour définition)
#     l1_ratio = (0.1, 0.25, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.99), 
#     # Compromise between L1 & L2 penalization chosen by cross validation
#     # L1_ratio set to 0 <=> Ridge régression
#     # L1_ratio set to 1 <=> Lasso régression
#     alphas = (0.001, 0.01, 0.02, 0.025, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0)
#     # List of alphas where to compute the models
#     # How much weight is given to each of L1/L2 penalities
# )

modele.fit (X_train, y_train)

#--Save the trained model to a file
with open(data_training_config.model_filepath, 'wb') as f:
    pickle.dump(modele, f)

from src.config import CONFIG_FILE_PATH
params_YAML = read_yaml(CONFIG_FILE_PATH)
with mlflow.start_run():
  mlflow.log_params(params_YAML)
  mlflow.sklearn.log_model(modele, artifact_path="model")

print("Model trained and saved successfully.")
