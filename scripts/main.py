# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the datasets
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data',
                 header=None, names=['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',
                                     'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
                                     'wheel-base', 'length', 'width', 'height', 'curb-weight',
                                     'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system',
                                     'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm',
                                     'city-mpg', 'highway-mpg', 'price'])

# Remove any rows with missing values
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Extract the relevant features and target variable
X = df[['engine-size', 'horsepower', 'curb-weight']]
y = df['price']

# Convert categorical variables into dummy variables
X = pd.get_dummies(X, columns=['num-of-doors', 'body-style', 'drive-wheels', 'fuel-system'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the prices on the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: ", mse)


