import numpy as np
from linear_regression import LinearRegression

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
print("predictions:", predictions)
