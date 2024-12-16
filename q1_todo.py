from utils import plot_data, generate_data
import numpy as np
import os

dir_working = "/media/adebayo/Windows/UofACanada/study/codes/CMPUT566_ML/coding_2/"

os.chdir(dir_working)


"""
Documentation:

Function generate() takes as input "A" or "B", it returns X, t.
X is two dimensional vectors, t is the list of labels (0 or 1).    

Function plot_data(X, t, w=None, bias=None, is_logistic=False, figure_name=None)
takes as input paris of (X, t) , parameter w, and bias. 
If you are plotting the decision boundary for a logistic classifier, set "is_logistic" as True
"figure_name" specifies the name of the saved diagram.
"""

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, t, lr=0.1, epochs=1000):
    """
    Given data, train your logistic classifier.
    Return weight and bias
    """

    n, d = X.shape
    w = np.zeros(d)
    b = 0

    for _ in range(epochs):
        z = np.dot(X, w) + b
        y = sigmoid(z)
        
        # Gradient computation
        error = y - t
        dw = np.dot(X.T, error) / n
        db = np.sum(error) / n

        # Update weights and bias
        w -= lr * dw
        b -= lr * db

    return w, b


def predict_logistic_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """

    z = np.dot(X, w) + b
    y = sigmoid(z)
    t = (y >= 0.5).astype(int)

    return t


def train_linear_regression(X, t):
    """
    Given data, train your linear regression classifier.
    Return weight and bias
    """

    n, d = X.shape
    X_aug = np.c_[np.ones((n, 1)), X]  # Add bias term
    theta = np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T @ t
    b = theta[0]
    w = theta[1:]


    return w, b


def predict_linear_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """

    y = np.dot(X, w) + b
    t = (y >= 0.5).astype(int)  # Threshold to classify
    
    return t


def get_accuracy(t, t_hat):
    """
    Calculate accuracy,
    """

    acc = np.mean(t == t_hat) * 100
    
    return acc


def main():
    # Dataset A
    # Linear regression classifier
    X, t = generate_data("A")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False,
              figure_name='dataset_A_linear.png')

    # logistic regression classifier
    X, t = generate_data("A")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of logistic regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True,
              figure_name='dataset_A_logistic.png')

    # Dataset B
    # Linear regression classifier
    X, t = generate_data("B")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False,
              figure_name='dataset_B_linear.png')

    # logistic regression classifier
    X, t = generate_data("B")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of logistic regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True,
              figure_name='dataset_B_logistic.png')


main()
