# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import warnings

warnings.filterwarnings("ignore")

# Load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
cols = [
    "symboling",
    "normalized-losses",
    "make",
    "fuel-type",
    "aspiration",
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-type",
    "num-of-cylinders",
    "engine-size",
    "fuel-system",
    "bore",
    "stroke",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price",
]
# df = pd.read_csv("./datasets/car-data.csv", names=cols, header=None)
df = pd.read_csv(url, names=cols, header=None)

# df.to_csv("./datasets/car-data.csv", index=False)

# Clean data
df = df.replace("?", np.nan)

# Convert categorical variables to numerical
df["num-of-doors"] = df["num-of-doors"].replace({"four": 4, "two": 2})
df["num-of-cylinders"] = df["num-of-cylinders"].replace(
    {"four": 4, "six": 6, "five": 5, "eight": 8, "two": 2, "twelve": 12, "three": 3}
)

encoder = OneHotEncoder(handle_unknown="ignore")
encoded_columns = [
    "fuel-type",
    "aspiration",
    "body-style",
    "drive-wheels",
    "engine-location",
    "engine-type",
    "num-of-cylinders",
    "fuel-system",
]
df_encoded = pd.DataFrame(
    encoder.fit_transform(df[encoded_columns]).toarray(),
    columns=encoder.get_feature_names(encoded_columns),
)
df.drop(encoded_columns, axis=1, inplace=True)
df = pd.concat([df, df_encoded], axis=1)

df = df.dropna()

# Split data into training and testing sets
# X = df.drop(["price", "make", "fuel-type", "aspiration", "body-style", "drive-wheels", "engine-location", "engine-type", "num-of-cylinders", "fuel-system"], axis=1)
X = df.drop(["price", "make"], axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Support Vector Regression
# svr = SVR(**{'C': 10, 'gamma': 'scale', 'kernel': 'linear'})
# # param_grid_svr = {'kernel': ['linear', 'poly', 'rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
# # grid_svr = GridSearchCV(svr, param_grid_svr, scoring='r2', cv=5, n_jobs=-1)
# # grid_svr.fit(X_train, y_train)
# svr.fit(X_train, y_train)
# # print("Best hyperparameters for SVR: ", grid_svr.best_params_)
# y_pred_svr = svr.predict(X_test)
# print("R-squared score for SVR: ", r2_score(y_test, y_pred_svr))

# Neural Network
# grid_nn = MLPRegressor(**{'activation': 'identity', 'alpha': 0.001, 'hidden_layer_sizes': (100,), 'solver': 'lbfgs'})
# param_grid_nn = {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['identity', 'logistic', 'tanh', 'relu'],
#  'solver': ['lbfgs', 'adam'], 'alpha': [0.0001, 0.001, 0.01]}
# grid_nn = GridSearchCV(nn, param_grid_nn, scoring='r2', cv=5, n_jobs=-1)
# grid_nn.fit(X_train, y_train)
# print("Best hyperparameters for Neural Network: ", grid_nn.best_params_)
# y_pred_nn = grid_nn.predict(X_test)
# print("R-squared score for Neural Network: ", r2_score(y_test, y_pred_nn))

grid_rf = RandomForestRegressor(
    **{
        "max_depth": None,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "n_estimators": 100,
    }
)
# param_grid_rf = {'n_estimators': [100, 300, 500], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10],
# 'min_samples_leaf': [1, 2, 4],}
# grid_rf = GridSearchCV(rf, param_grid_rf, scoring='r2', cv=5, n_jobs=-1)
grid_rf.fit(X_train, y_train)
# print("Best hyperparameters for Random Forest: ", grid_rf.best_params_)
y_pred_rf = grid_rf.predict(X_test)
print("R-squared score for Random Forest: ", r2_score(y_test, y_pred_rf))
