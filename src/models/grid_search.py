import pandas as pd 
from sklearn import ensemble
import numpy as np
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config_manager import ConfigurationManager

config_manager = ConfigurationManager()

data_split_config = config_manager.get_data_split_config()
data_grid_search_config = config_manager.get_data_grid_search_config()

X_train = pd.read_csv(data_split_config.output_folderpath 
                      + '/' + data_split_config.output_X_train_scaled_filename)
y_train = pd.read_csv(data_split_config.output_folderpath 
                      + '/' + data_split_config.output_y_train_filename)
y_train = np.ravel(y_train)

# from sklearn.ensemble import GradientBoostingRegressor
# modele = GradientBoostingRegressor (
#     n_estimators = 1000,
#     max_depth = 10000,
#     max_features = 15
# )

from sklearn.ensemble import RandomForestRegressor
modele = RandomForestRegressor ()

print ("Paramètre du modèle", modele.get_params().keys())

params = {
    'n_estimators': data_grid_search_config.n_estimators_range,
    'max_depth': data_grid_search_config.max_depth_range
    #,
    #'max_features' : [ 5, 10, 15]
}

from sklearn.model_selection import GridSearchCV
grid_modele = GridSearchCV(estimator = modele, 
    param_grid = params, 
    scoring = None, # métrique par défaut : accuracy
    cv = 3, # Cross validation : Nombre de folds
    verbose = 2
) 

grid_modele = grid_modele.fit(X_train, y_train)

#--Save the best params to a file
with open(data_grid_search_config.output_filepath, 'wb') as f:
    pickle.dump(grid_modele.best_params_, f)
print("Grid search - best params saved successfully.")