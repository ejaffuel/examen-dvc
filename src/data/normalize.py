from sklearn import preprocessing
import pandas as pd 
from check_structure import check_existing_file

X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')

scaler = preprocessing.StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index = X_train.index)
X_train_scaled.columns = X_train.columns
X_test_scaled = pd.DataFrame(scaler.transform(X_test), index = X_test.index)
X_test_scaled.columns = X_test.columns

def save_dataframe(dataframe, output_filepath):
    if check_existing_file(output_filepath):
        dataframe.to_csv(output_filepath, index=False)

save_dataframe(X_train_scaled, "data/processed/X_train_scaled.csv")
save_dataframe(X_test_scaled, "data/processed/X_test_scaled.csv")