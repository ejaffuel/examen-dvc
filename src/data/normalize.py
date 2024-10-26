from sklearn import preprocessing
import pandas as pd 
from check_structure import check_existing_file
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config_manager import ConfigurationManager

config_manager = ConfigurationManager()
data_split_config = config_manager.get_data_split_config()

X_train = pd.read_csv(data_split_config.output_folderpath
                      +data_split_config.X_train_filename)
X_test = pd.read_csv(data_split_config.output_folderpath
                      +data_split_config.X_test_filename)

scaler = preprocessing.StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index = X_train.index)
X_train_scaled.columns = X_train.columns
X_test_scaled = pd.DataFrame(scaler.transform(X_test), index = X_test.index)
X_test_scaled.columns = X_test.columns

def save_dataframe(dataframe, output_filepath):
    if check_existing_file(output_filepath):
        dataframe.to_csv(output_filepath, index=False)

save_dataframe(X_train_scaled, 
               data_split_config.output_folderpath
               +data_split_config.X_train_scaled_filename)
save_dataframe(X_test_scaled, 
               data_split_config.output_folderpath
               +data_split_config.X_test_scaled_filename)