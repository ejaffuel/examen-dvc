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

    X_train_filename: Path
    X_train_scaled_filename: Path

    X_test_filename: Path
    X_test_scaled_filename: Path

    y_train_filename: Path
    y_test_filename: Path

@dataclass(frozen=True)
class Data_grid_search_config:
    best_params_filepath : Path
    n_estimators_range: list
    max_depth_range: list

@dataclass(frozen=True)
class Data_training_config:
    model_filepath : Path

