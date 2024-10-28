# Examen DVC and Dagshub
- **Author**: Eddie JAFFUEL (sep24_continu_mlops)
- **Delivery**: Branch "**Master**" / 27 oct 2024
- **Github repository**: https://github.com/ejaffuel/examen-dvc
- **DagsHub repository**: https://dagshub.com/ejaffuel/examen-dvc
---
## Installation guide
- Create and activate your virtual environment and install packages with:
    ```sh
    pip install -r requirements.txt
    ```
---
## User guide
- Launch pipeline with 
    ```sh
    dvc repro
    ```
- Pipeline parameters : ![DVC Pipeline Configuration YAML](./params.yaml)
    - Tous les chemins et noms de fichiers sont paramètrables pour toutes les étapes
    - Pour l'étape "split": le parametre "test_size" a été défini (taux de la partie test)
    - Pour l'étape "gridSearch" : les ranges des 2 hyperparametres sont paramétrables
- Pipeline definition : ![DVC Pipeline Definition YAML](./dvc.yaml)
    - La pipeline s'appuie sur la configuration des paramètres 
    - :warning: La pipeline n'est pas exactement celle de l'énoncé, voici les différences:
        - Pour importer raw.csv "external file" avec URL: j'ai utilisé "import_raw_data.py" dans l'étape "Split"
        - Toutes les scripts de la pipeline dépendent de "params.yaml"
- Pipeline visualization : ![DVC Pipeline Graphical](docs/Examen_DVC_Pipeline.png)
---
## Tree of Folders and files (managed under **Git**)
```git ls-tree -r --name-only HEAD | tree --fromfile```
```
├── .dvc
│   └── config
├── .dvcignore
├── .gitignore
├── requirements.txt    The requirements of the package to install
├── setup.py            To manage package dependency import

├── README.md           The main documentation 
├── docs                Other elements used in documentation

├── dvc.yaml            The configuration of the pipeline
├── params.yaml         The parameters of the pipeline
├── dvc.lock            The result of pipeline execution

├── data                DVC Data folder
│   ├── raw_data            Original dataset with label
│   ├── processed           Used to store processed dataset split in (X|y)_(train|test)
│   └── predict             Used to store predictions
├── models              DVC Output Model data folder (best_params, training model)
├── metrics             DVC Output Folder for metric performance (scores)

└── src                     The Source code folder (contain all scripts)
    ├── common_utils.py         several utility functions

    ├── config.py               Path to configuration YAML file (params.YAML)
    ├── entity.py               Mirror classes for configuration YAML file
    ├── config_manager.py       Mapping between Mirror classes and configuration YAML file

    ├── data                    Scrits to manage the processed data
    │   ├── check_structure.py      Utilities functions
    │   ├── import_raw_data.py      Step1: import raw data from distant repository
    │   ├── data_split.py           Step2: split raw data in (X|y)_(train|test)
    │   └── normalize.py            Step3: normalize X_(train|test) => X_(train|X_test)_scaled

    └── models                  Scripts to manage the model training and evaluation
        ├── grid_search.py          Step4: Perform a Grid Search to find best hyperparameters
        ├── training.py             Step5: Train the model with best hyperparameters
        └── evaluate.py             Step6: Evaluate the model
```
