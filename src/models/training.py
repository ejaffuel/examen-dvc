
import sklearn
import pandas as pd 
from sklearn import ensemble
import pickle
import numpy as np

X_train = pd.read_csv('data/processed/X_train_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv')
y_train = np.ravel(y_train)

# from sklearn.linear_model import LinearRegression
# modele = LinearRegression()

from sklearn.ensemble import GradientBoostingRegressor
modele = GradientBoostingRegressor (
    n_estimators = 1000,
    max_depth = 10000,
    max_features = 15
)

modele.fit (X_train, y_train)

#--Save the trained model to a file
model_filename = './models/gbr_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(modele, f)
print("Model trained and saved successfully.")
