import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from check_structure import check_existing_file, check_existing_folder
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config_manager import ConfigurationManager
from src.data.import_raw_data import import_raw_data

config_manager = ConfigurationManager()
data_import_raw_config = config_manager.get_data_import_raw_config()

data_split_config = config_manager.get_data_split_config()

def main(input_url, input_filepath, output_folderpath):
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    process_data(input_url, input_filepath, output_folderpath)

def process_data(input_url, input_filepath, output_folderpath):
    # Import datasets
    df = import_dataset(input_url, input_filepath, sep=",")

    # Ne pas considérer les dates (d'après l'énoncé)
    df = df.drop(['date'], axis=1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)

    # Create folder if necessary
    create_folder_if_necessary(output_folderpath)

    # Save dataframes to their respective output file paths
    save_dataframes(X_train, X_test, y_train, y_test, output_folderpath)

def import_dataset(input_url, file_path, **kwargs):

    output_folderpath = Path(file_path).parent
    output_filename = Path(file_path).name
    import_raw_data(input_url, output_folderpath, output_filename)
    return pd.read_csv(file_path, **kwargs)

def split_data(df):
    # Split data into training and testing sets
    target = df['silica_concentrate']
    feats = df.drop(['silica_concentrate'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(feats, 
                                                        target, 
                                                        test_size=data_split_config.test_size, 
                                                        random_state=42)
    return X_train, X_test, y_train, y_test

def create_folder_if_necessary(output_folderpath):
    # Create folder if necessary
    if check_existing_folder(output_folderpath):
        os.makedirs(output_folderpath)

def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, 
                               X_test, 
                               y_train, 
                               y_test], 
                              [data_split_config.X_train_filename, 
                               data_split_config.X_test_filename,
                               data_split_config.y_train_filename,
                               data_split_config.y_test_filename]):
        output_filepath = os.path.join(data_split_config.output_folderpath, 
                                       filename 
                                      )
        #if check_existing_file(output_filepath):
        file.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    input_url = data_import_raw_config.input_url
    input_filepath = data_import_raw_config.output_filepath
    output_folderpath = data_split_config.output_folderpath
    main(input_url, input_filepath, output_folderpath)