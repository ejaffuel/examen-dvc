
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

# from sklearn.ensemble import GradientBoostingRegressor
# modele = GradientBoostingRegressor (
#     n_estimators = 1000,
#     max_depth = 10000,
#     max_features = 15
# )

from sklearn.linear_model import ElasticNetCV
modele = ElasticNetCV(
    cv = 8, # cross validation with 8-folds (Voir 2.7.9 pour définition)
    l1_ratio = (0.1, 0.25, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.99), 
    # Compromise between L1 & L2 penalization chosen by cross validation
    # L1_ratio set to 0 <=> Ridge régression
    # L1_ratio set to 1 <=> Lasso régression
    alphas = (0.001, 0.01, 0.02, 0.025, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0)
    # List of alphas where to compute the models
    # How much weight is given to each of L1/L2 penalities
)

modele.fit (X_train, y_train)

#--Save the trained model to a file
model_filename = './models/gbr_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(modele, f)
print("Model trained and saved successfully.")
