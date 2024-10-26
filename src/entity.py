from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Data_import_raw_config:
    input_url: str
    output_filepath : Path

@dataclass(frozen=True)
class Data_split_config:
    test_size: float
    
    output_folderpath:  Path

    output_X_train_filename: Path
    output_X_train_scaled_filename: Path

    output_X_test_filename: Path
    output_X_test_scaled_filename: Path

    output_y_train_filename: Path
    output_y_test_filename: Path

@dataclass(frozen=True)
class Data_grid_search_config:
    output_filepath : Path
    n_estimators_range: list
    max_depth_range: list

@dataclass(frozen=True)
class Data_training_config:
    output_filepath : Path

