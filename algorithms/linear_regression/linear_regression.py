import numpy as np


class LinearRegression:
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        X_augmented = np.column_stack((np.ones(X.shape[0]), X))
        self.theta = np.linalg.lstsq(
            X_augmented.T @ X_augmented, X_augmented.T @ y, rcond=None
        )[0]

    def predict(self, X):
        X_augmented = np.column_stack((np.ones(X.shape[0]), X))
        y_predict = X_augmented @ self.theta
        return y_predict

    def score(self, X, y, treshold=0.5):
        y_predict = self.predict(X)
        return np.mean(np.abs(y_predict - y) <= treshold)
