#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os

dir_working = "/media/adebayo/Windows/UofACanada/study/codes/CMPUT566_ML/"

os.chdir(dir_working)

def rotate(data, degree):
    # data: M x 2
    theta = np.pi/180 * degree
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])  # rotation matrix
    return np.dot(data, R.T)


def leastSquares(X, Y):
    # In this function, X is always the input, Y is always the output
    # X: M x (d+1), Y: M x 1, where d=1 here
    # return weights w

    # TODO: YOUR CODE HERE
    # closed form solution by matrix-vector representations only
    w = np.linalg.inv(X.T @ X) @ X.T @ Y

    return w


def model(X, w):
    # X: M x (d+1)
    # w: d+1
    # return y_hat: M x 1

    # TODO: YOUR CODE HERE
    y_hat = X @ w

    return y_hat


def generate_data(M, var1, var2, degree):

    # data generate involves two steps:
    # Step I: generating 2-D data, where two axis are independent
    # M (scalar): The number of data samples
    # var1 (scalar): variance of a
    # var2 (scalar): variance of b

    mu = [0, 0]

    Cov = [[var1, 0],
           [0,  var2]]

    data = np.random.multivariate_normal(mu, Cov, M)
    # shape: M x 2

    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], color="blue")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('a')
    plt.ylabel('b')
    plt.tight_layout()
    plt.savefig('data_ab_'+str(var2)+'.jpg')
    plt.close()  # Close to prevent display

    # Step II: rotate data by 45 degree counter-clockwise,
    # so that the two dimensions are in fact correlated

    data = rotate(data, degree)
    plt.tight_layout()
    plt.figure()
    # plot the data points
    plt.scatter(data[:, 0], data[:, 1], color="blue")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"Rotated Data (x, y) with var2={var2} and degree={degree}")

    # plot the line where data are mostly generated around
    X_new = np.linspace(-5, 5, 100, endpoint=True).reshape([100, 1])

    Y_new = np.tan(np.pi/180*degree)*X_new
    plt.plot(X_new, Y_new, color="blue", linestyle='dashed')
    plt.tight_layout()
    plt.savefig('data_xy_'+str(var2) + '_' + str(degree) + '.jpg')
    plt.close()  # Close to prevent display
    return data


###########################
# Main code starts here
###########################
# Settings
M = 5000
var1 = 1
# Function to plot the regression models
import pandas as pd

# Function to track the regression model parameters and print in tabular form
def plot_regression_models_and_print_table(var2_list, degrees_list, M=5000, var1=1):
    # List to store the results for the table
    results = []

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    
    # First row: varying var2 with degree fixed at 45
    for i, var2 in enumerate(var2_list):
        # Generate data for each var2
        data = generate_data(M, var1, var2, 45)  # degree is fixed to 45 for var2 variation
        
        # Training the linear regression model predicting y from x (x2y)
        Input = data[:, 0].reshape((-1, 1))  # M x d, where d=1
        Input_aug = np.concatenate([Input, np.ones([M, 1])], axis=1)  # M x (d+1) augmented feature
        Output = data[:, 1].reshape((-1, 1))  # M x 1
        w_x2y = leastSquares(Input_aug, Output)  # (d+1) x 1, where d=1

        # Training the linear regression model predicting x from y (y2x)
        Input = data[:, 1].reshape((-1, 1))  # M x d, where d=1
        Input_aug = np.concatenate([Input, np.ones([M, 1])], axis=1)  # M x (d+1) augmented feature
        Output = data[:, 0].reshape((-1, 1))  # M x 1
        w_y2x = leastSquares(Input_aug, Output)  # (d+1) x 1, where d=1

        # Store the weights and biases in the results list
        results.append({
            'var2': var2,
            'degree': 45,
            'w_x2y (weight)': w_x2y[0, 0],
            'w_x2y (bias)': w_x2y[1, 0],
            'w_y2x (weight)': w_y2x[0, 0],
            'w_y2x (bias)': w_y2x[1, 0]
        })

        # Plotting in the first row (varying var2)
        ax = axs[0, i]
        X = data[:, 0].reshape((-1, 1))  # M x d, where d=1
        Y = data[:, 1].reshape((-1, 1))  # M x d, where d=1
        ax.scatter(X, Y, color="blue", marker='x')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # plot the line where data are mostly generated around
        X_new = np.linspace(-4, 4, 100, endpoint=True).reshape([100, 1])
        Y_new = np.tan(np.pi/180*45)*X_new
        ax.plot(X_new, Y_new, color="blue", linestyle='dashed')

        # Plot the prediction of y from x (x2y)
        #X_new = np.linspace(-4, 4, 100, endpoint=True).reshape([100, 1])
        X_new_aug = np.concatenate([X_new, np.ones([X_new.shape[0], 1])], axis=1)
        ax.plot(X_new, model(X_new_aug, w_x2y), color="red", label="x2y")

        # Plot the prediction of x from y (y2x)
        Y_new = np.linspace(-4, 4, 100, endpoint=True).reshape([100, 1])
        Y_new_aug = np.concatenate([Y_new, np.ones([Y_new.shape[0], 1])], axis=1)
        ax.plot(model(Y_new_aug, w_y2x), Y_new, color="green", label="y2x")
        ax.legend()
        ax.set_title(f"var2 = {var2}")
    
    # Second row: varying degree with var2 fixed at 0.1
    for i, degree in enumerate(degrees_list):
        # Generate data for each degree (var2 is fixed to 0.1 for degree variation)
        data = generate_data(M, var1, 0.1, degree)
        
        # Training the linear regression model predicting y from x (x2y)
        Input = data[:, 0].reshape((-1, 1))  # M x d, where d=1
        Input_aug = np.concatenate([Input, np.ones([M, 1])], axis=1)  # M x (d+1) augmented feature
        Output = data[:, 1].reshape((-1, 1))  # M x 1
        w_x2y = leastSquares(Input_aug, Output)  # (d+1) x 1, where d=1

        # Training the linear regression model predicting x from y (y2x)
        Input = data[:, 1].reshape((-1, 1))  # M x d, where d=1
        Input_aug = np.concatenate([Input, np.ones([M, 1])], axis=1)  # M x (d+1) augmented feature
        Output = data[:, 0].reshape((-1, 1))  # M x 1
        w_y2x = leastSquares(Input_aug, Output)  # (d+1) x 1, where d=1

        # Store the weights and biases in the results list
        results.append({
            'var2': 0.1,
            'degree': degree,
            'w_x2y (weight)': w_x2y[0, 0],
            'w_x2y (bias)': w_x2y[1, 0],
            'w_y2x (weight)': w_y2x[0, 0],
            'w_y2x (bias)': w_y2x[1, 0]
        })

        # Plotting in the second row (varying degree)
        ax = axs[1, i]
        X = data[:, 0].reshape((-1, 1))  # M x d, where d=1
        Y = data[:, 1].reshape((-1, 1))  # M x d, where d=1
        ax.scatter(X, Y, color="blue", marker='x')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xlabel('x')
        ax.set_ylabel('y')


        # plot the line where data are mostly generated around
        X_new = np.linspace(-4, 4, 100, endpoint=True).reshape([100, 1])
        Y_new = np.tan(np.pi/180*degree)*X_new
        ax.plot(X_new, Y_new, color="blue", linestyle='dashed')

        # Plot the prediction of y from x (x2y)
        #X_new = np.linspace(-4, 4, 100, endpoint=True).reshape([100, 1])
        X_new_aug = np.concatenate([X_new, np.ones([X_new.shape[0], 1])], axis=1)
        ax.plot(X_new, model(X_new_aug, w_x2y), color="red", label="x2y")

        # Plot the prediction of x from y (y2x)
        Y_new = np.linspace(-4, 4, 100, endpoint=True).reshape([100, 1])
        Y_new_aug = np.concatenate([Y_new, np.ones([Y_new.shape[0], 1])], axis=1)
        ax.plot(model(Y_new_aug, w_y2x), Y_new, color="green", label="y2x")
        ax.legend()
        ax.set_title(f"Degree = {degree}")

    plt.tight_layout()
    plt.savefig('Regression_model_var2_degree.jpg')
    plt.show()

    # Convert results to a pandas DataFrame and print the table
    df_results = pd.DataFrame(results)
    print(df_results)

# Example usage
var2_list = [0.1, 0.3, 0.8]
degrees_list = [30, 45, 60]
plot_regression_models_and_print_table(var2_list, degrees_list)


