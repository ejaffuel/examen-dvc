import pandas as pd
import numpy as np
from pathlib import Path
import click
import logging
from sklearn.model_selection import train_test_split
from check_structure import check_existing_file, check_existing_folder
import os

input_filepath = "data/raw_data/raw.csv"
output_filepath = "data/processed"

def main(input_filepath, output_filepath):
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    process_data(input_filepath)

def process_data(input_filepath):
    # Import datasets
    df = import_dataset(input_filepath, sep=",")

    # Ne pas considérer les dates (d'après l'énoncé)
    df = df.drop(['date'], axis=1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)

    # Create folder if necessary
    create_folder_if_necessary(output_filepath)

    # Save dataframes to their respective output file paths
    save_dataframes(X_train, X_test, y_train, y_test, output_filepath)

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

def split_data(df):
    # Split data into training and testing sets
    target = df['silica_concentrate']
    feats = df.drop(['silica_concentrate'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def create_folder_if_necessary(output_folderpath):
    # Create folder if necessary
    if check_existing_folder(output_folderpath):
        os.makedirs(output_folderpath)

def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main(input_filepath, output_filepath)