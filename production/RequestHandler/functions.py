import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()


def random_forest(data):
    """Processes prediction calls to random forest
    Args:
        data (int): value to predict
    Returns:
        bool, ndarray: predicted value
    """
    # locate model artifact
    with open("./artifacts/rf_model.pkl", "rb") as model_file:
        # load model artifact
        rf_model = pickle.load(model_file)

    # make predictions
    data = rf_model.predict(scalar.fit_transform(np.array(data).reshape(1, -1)))

    # return status and predicted data
    return True, data


def decision_tree(data):
    """Processes prediction calls to decision tree
    Args:
        data (int): value to predict
    Returns:
        bool, ndarray: predicted value
    """
    # locate model artifact
    with open("./artifacts/dt_model.pkl", "rb") as model_file:
        # load model artifact
        dt_model = pickle.load(model_file)

    # make predictions
    data = dt_model.predict(scalar.fit_transform(np.array(data).reshape(1, -1)))

    # return status and predicted data
    return True, data


def xgboost(data):
    """Processes prediction calls to decision tree
    Args:
        data (int): value to predict
    Returns:
        bool, ndarray: predicted value
    """
    # locate model artifact
    with open("./artifacts/xgb_model.pkl", "rb") as model_file:
        # load model artifact
        xgb_model = pickle.load(model_file)

    # make predictions
    data = xgb_model.predict(scalar.fit_transform(np.array(data).reshape(1, -1)))

    # return status and predicted data
    return True, data
