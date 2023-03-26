# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

# Load the dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data', header=None)

# Add column names to the dataset
headers = ['symboling', 'normalized_losses', 'make', 'fuel_type', 'aspiration', 'num_doors', 'body_style', 'drive_wheels', 'engine_location', 'wheel_base', 'length', 'width', 'height', 'curb_weight', 'engine_type', 'num_cylinders', 'engine_size', 'fuel_system', 'bore', 'stroke', 'compression_ratio', 'horsepower', 'peak_rpm', 'city_mpg', 'highway_mpg', 'price']
df.columns = headers

# Replace missing values with NaN
df.replace('?', np.nan, inplace=True)

# Drop any rows with missing values
df.dropna(subset=['price'], inplace=True)

# Convert numeric columns to float data type
df['normalized_losses'] = df['normalized_losses'].astype('float')
df['bore'] = df['bore'].astype('float')
df['stroke'] = df['stroke'].astype('float')
df['horsepower'] = df['horsepower'].astype('float')
df['peak_rpm'] = df['peak_rpm'].astype('float')
df['price'] = df['price'].astype('float')

# Extract the relevant features and target variable
X = df[['horsepower', 'curb_weight', 'engine_size', 'highway_mpg']]
y = df['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train the linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Support Vector Regression
svr = SVR()
param_grid_svr = {'kernel': ['linear', 'poly', 'rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
grid_svr = GridSearchCV(svr, param_grid_svr, scoring='r2', cv=5, n_jobs=-1)
grid_svr.fit(X_train, y_train)
print("Best hyperparameters for SVR: ", grid_svr.best_params_)
y_pred_svr = grid_svr.predict(X_test)
print("R-squared score for SVR: ", r2_score(y_test, y_pred_svr))

# Predict the prices on the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: ", mse)

