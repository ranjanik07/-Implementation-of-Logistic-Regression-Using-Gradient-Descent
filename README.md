# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize weights and bias.
2. Compute predictions using sigmoid.
3. Update weights and bias with gradient descent.
4. Classify output (>0.5 = Placed, else Not Placed).

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: RANJANI K
RegisterNumber:  212224230220

```
```
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    for _ in range(epochs):
        linear_model = np.dot(X, weights) + bias
        y_pred = sigmoid(linear_model)
        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)
        weights -= lr * dw
        bias -= lr * db
    return weights, bias

def predict(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_model)
    return [1 if i > 0.5 else 0 for i in y_pred]

X = np.array([
    [8.5, 8, 1],
    [7.2, 6, 0],
    [6.8, 5, 0],
    [8.9, 9, 1],
    [5.5, 4, 0],
    [6.0, 5, 0],
    [7.8, 7, 1],
    [8.0, 8, 1],
    [7.0, 6, 0],
    [5.8, 3, 0]
])
y = np.array([1,1,0,1,0,0,1,1,0,0])

weights, bias = logistic_regression(X, y, lr=0.01, epochs=10000)
y_pred = predict(X, weights, bias)
accuracy = np.mean(y_pred == y)
print("Accuracy:", accuracy)

new_student = np.array([[7.5, 7, 1]])
prediction = predict(new_student, weights, bias)
print("New Student Placement Prediction:", "Placed" if prediction[0]==1 else "Not Placed")
```
## Output:
<img width="506" height="67" alt="Screenshot 2025-09-29 193505" src="https://github.com/user-attachments/assets/4b62a20d-6b25-47f3-b8fa-1b86ef06f04b" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

