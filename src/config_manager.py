import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import CONFIG_FILE_PATH
from src.common_utils import read_yaml, create_directories
from src.entity import (Data_import_raw_config,
                        Data_split_config, 
                        Data_grid_search_config, 
                        Data_training_config)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)

    def get_data_import_raw_config(self) -> Data_import_raw_config:
          config = self.config.import_raw_config

          create_directories([Path(config.output_filepath).parent])

          import_raw_config = Data_import_raw_config(
                  input_url = config.input_url,
                  output_filepath = config.output_filepath
          )
          return import_raw_config


    def get_data_split_config(self) -> Data_split_config:
          config = self.config.split_config

          create_directories([config.output_folderpath])

          data_split_config = Data_split_config(
                  test_size = config.test_size,

                  output_folderpath = config.output_folderpath,

                  output_X_train_filename = config.output_X_train_filename,
                  output_X_train_scaled_filename = config.output_X_train_scaled_filename,

                  output_X_test_filename = config.output_X_test_filename,
                  output_X_test_scaled_filename = config.output_X_test_scaled_filename,

                  output_y_train_filename = config.output_y_train_filename,
                  output_y_test_filename = config.output_y_test_filename
          )
          return data_split_config

    def get_data_grid_search_config(self) -> Data_grid_search_config:
          config = self.config.grid_search_config

          create_directories([Path(config.output_filepath).parent])

          data_grid_search_config = Data_grid_search_config(
                output_filepath = config.output_filepath,
                n_estimators_range = config.n_estimators_range,
                max_depth_range = config.max_depth_range
          )
          return data_grid_search_config

    def get_data_training_config(self) -> Data_training_config:
          config = self.config.training_config

          create_directories([Path(config.output_filepath).parent])

          data_training_config = Data_training_config(
                n_estimators_range = config.n_estimators_range,
                max_depth_range = config.max_depth_range
          )
          return data_training_config
