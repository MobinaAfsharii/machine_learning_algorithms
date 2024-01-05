# Linear Regression

### Overview
This [\(linear_regression.py\)](./linear_regression.py) Python code provides a simple implementation of linear regression using the method of least squares. Linear regression is a supervised learning algorithm used for predicting a continuous target variable based on one or more independent features.

### Class: `LinearRegression`

#### Initialization
```python
class LinearRegression:
    def __init__(self):
        self.theta = None
```
The class initializes with an empty `theta` attribute, which will eventually store the coefficients of the linear regression model.

#### Method: `fit(X, y)`
```python
    def fit(self, X, y):
        X_augmented = np.column_stack((np.ones(X.shape[0]), X))
        self.theta = np.linalg.lstsq(
            X_augmented.T @ X_augmented, X_augmented.T @ y, rcond=None
        )[0]
```
This method fits the linear regression model to the input data (`X` and `y`). It adds a column of ones to the input matrix `X` to account for the intercept term. The method then computes the coefficients (`theta`) using the least squares solution provided by `numpy.linalg.lstsq`.

#### Method: `predict(X)`
```python
    def predict(self, X):
        X_augmented = np.column_stack((np.ones(X.shape[0]), X))
        y_predict = X_augmented @ self.theta
        return y_predict
```
The `predict` method takes input features `X` and predicts the corresponding target variable using the learned coefficients. Again, it augments the input matrix with a column of ones before computing the predicted values.

#### Method: `score(X, y, threshold=0.5)`
```python
    def score(self, X, y, threshold=0.5):
        y_predict = self.predict(X)
        return np.mean(np.abs(y_predict - y) <= threshold)
```
The `score` method evaluates the accuracy of the model predictions by comparing them to the actual target values. It uses a specified threshold (default is 0.5) to determine if the predicted values are correct. The score is calculated as the mean of the absolute differences between predicted and actual values within the threshold.

### Mathematical Concepts

#### Linear Regression
Linear regression models the relationship between a dependent variable (`y`) and one or more independent variables (`X`) by fitting a linear equation. The goal is to find the coefficients (`theta`) that minimize the sum of squared differences between the predicted and actual values.

#### Least Squares Solution
The least squares solution is a method for finding the coefficients (`theta`) that minimize the sum of the squared residuals. In this implementation, it is computed using the `numpy.linalg.lstsq` function.

### Usage Example

you can find full example at [./linear_regression.ipynb](./linear_regression.ipynb)


```python
import numpy as np

# Sample data
X_train = np.array([[1], [2], [3]])
y_train = np.array([2, 3, 4])

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
X_test = np.array([[4], [5]])
predictions = model.predict(X_test)

# Evaluate the model
accuracy = model.score(X_test, np.array([5, 6]))
print(f"Model Accuracy: {accuracy}")
```

Feel free to customize the above explanation based on the intended audience and level of understanding.