#!/usr/bin/env python3

import pickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os

dir_working = "/media/adebayo/Windows/UofACanada/study/codes/CMPUT566_ML/"

os.chdir(dir_working)


# Define a function to extend features by adding quadratic terms
def extend_features(X):
    X_quad = np.hstack([X, X[:, 1:] ** 2])  # Add quadratic terms, excluding bias term
    return X_quad


def predict(X, w, y=None):
    # X_new: Nsample x (d+1)
    # w: (d+1) x 1
    # y_new: Nsample

    # TODO: Your code here

    y_hat_norm = X @ w  # Prediction using the weight vector (normalized space)

    # Unnormalize predictions back to the original scale
    y_hat_orig = y_hat_norm * std_y + mean_y

    # If true labels (normalized) are provided, calculate loss and risk
    if y is not None:
        # Mean squared error for normalized loss
        loss = np.mean((y_hat_norm - y) ** 2)

        # Mean absolute error (risk) calculated in original space
        y_orig = y * std_y + mean_y  # Unnormalize the target values
        risk = np.mean(np.abs(y_hat_orig - y_orig))  # Risk is calculated in the original scale
    else:
        loss = None
        risk = None
    
    y_hat = y_hat_norm

    return y_hat, loss, risk


# Store performance data for plotting
performance_data = []

def train_with_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    global performance_data
    best_risk_val = float('inf')
    best_hyperparameters = None
    best_w = None
    best_losses_train = None
    best_risks_val = None

    # Loop through each combination of hyperparameters
    for lambda_decay in lambda_candidates:
        for alpha_value in alpha_values:
            for batch_size_value in batch_size_values:
                for MaxIter_value in MaxIter_values:
                    print(f"Training with lambda_decay={lambda_decay}, alpha={alpha_value}, batch_size={batch_size_value}, MaxIter={MaxIter_value}")
                    
                    decay = lambda_decay
                    alpha = alpha_value
                    batch_size = batch_size_value
                    MaxIter = MaxIter_value

                    w = np.zeros([X_train.shape[1], 1])  # Initialize weights
                    losses_train = []
                    risks_val = []

                    w_best = None
                    risk_best = 10000
                    epoch_best = 0

                    for epoch in range(MaxIter):
                        loss_this_epoch = 0
                        
                        # Mini-batch gradient descent
                        for b in range(int(np.ceil(X_train.shape[0] / batch_size))):
                            X_batch = X_train[b * batch_size: (b + 1) * batch_size]
                            y_batch = y_train[b * batch_size: (b + 1) * batch_size]

                            y_hat_batch, loss_batch, _ = predict(X_batch, w, y_batch)
                            loss_this_epoch += loss_batch

                            # Gradient computation and weight update
                            gradient = (2 / batch_size) * X_batch.T @ (y_hat_batch - y_batch) + 2 * decay * w
                            w -= alpha * gradient  # Update weights using gradient descent

                        # 1. Compute the training loss by averaging loss_this_epoch
                        avg_loss_train = loss_this_epoch / X_train.shape[0]
                        losses_train.append(avg_loss_train)

                        # 2. Perform validation on the validation set by the risk
                        _, _, risk_val = predict(X_val, w, y_val.reshape(-1, 1))
                        risks_val.append(risk_val)

                        # 3. Track the best validation risk and weights
                        if risk_val < risk_best:
                            risk_best = risk_val
                            w_best = w.copy()
                            epoch_best = epoch
                    # Save the performance data for heatmap
                    performance_data.append({
                        'lambda_decay': lambda_decay,
                        'alpha': alpha_value,
                        'batch_size': batch_size_value,
                        'MaxIter': MaxIter_value,
                        'validation_risk': risk_best
                    })
                    # Print the best stats for each hyperparameter combination
                    print(f"Best validation risk {risk_best} at epoch {epoch_best} for lambda_decay={lambda_decay}, alpha={alpha_value}, batch_size={batch_size_value}, MaxIter={MaxIter_value}")

                    # Find the best epoch with minimum validation risk
                    min_risk_val = min(risks_val)
                    min_epoch = risks_val.index(min_risk_val)

                    # If this hyperparameter combination is better, update the best parameters and track losses and risks
                    if min_risk_val < best_risk_val:
                        best_risk_val = min_risk_val
                        best_hyperparameters = {
                            'lambda_decay': lambda_decay,
                            'alpha': alpha_value,
                            'batch_size': batch_size_value,
                            'MaxIter': MaxIter_value
                        }
                        best_epoch = min_epoch
                        best_w = w_best
                        best_losses_train = losses_train  # Track the best losses for plotting
                        best_risks_val = risks_val  # Track the best risks for plotting

    # Print the best combination of hyperparameters
    print(f"\nBest hyperparameters: {best_hyperparameters}")
    print(f"Best epoch = {best_epoch + 1}")
    print(f"Validation risk (MAE) = {best_risk_val}")

    return best_w, best_hyperparameters, best_epoch, best_risk_val, best_losses_train, best_risks_val



############################
# Main code starts here
############################
# Load data
with open("housing.pkl", "rb") as f:
    (X, y) = pickle.load(f)

# X: sample x dimension
# y: sample x 1

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# Augment feature
X_ = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)
# X_: Nsample x (d+1)

# normalize features:
mean_y = np.mean(y)
std_y = np.std(y)

y = (y - np.mean(y)) / np.std(y)

# print(X.shape, y.shape) # It's always helpful to print the shape of a variable


# Randomly shuffle the data
np.random.seed(314)
np.random.shuffle(X_)
np.random.seed(314)
np.random.shuffle(y)

X_train = X_[:300]
y_train = y[:300]

X_val = X_[300:400]
y_val = y[300:400]

X_test = X_[400:]
y_test = y[400:]


# Extend the features with quadratic terms
X_train_ext = extend_features(X_train)
X_val_ext = extend_features(X_val)
X_test_ext = extend_features(X_test)

#####################
# Global hyperparameters declared outside the function
alpha_values = [0.01, 0.001, 0.0001]  # learning rate candidates
batch_size_values = [16, 32, 64]  # batch size candidates
MaxIter_values = [50, 100, 200]  # Maximum iteration candidates
lambda_candidates = [3, 1, 0.3, 0.1, 0.03, 0.01]  # lambda decay candidates


# TODO: Your code here
w_best, best_hyperparameters, best_epoch, best_risk_val, best_losses_train, best_risks_val = train_with_hyperparameter_tuning(
    X_train_ext, y_train, X_val_ext, y_val
)

# Perform test by the weights yielding the best validation performance

# Report numbers and draw plots as required.
y_hat_test, _, risk_test = predict(X_test_ext, w_best, y_test.reshape(-1, 1))
print(f"Test risk (absolute error): {risk_test}")




# Define a plotting function to show the learning curves and heatmap

def plot_learning_curves(losses_train, risks_val, performance_data):
    epochs = range(1, len(losses_train) + 1)
    
    # Create a larger figure to accommodate both learning curves and the heatmap
    plt.figure(figsize=(10, 6))

    # Training loss curve (left on the first row)
    plt.subplot(2, 2, 1)
    plt.plot(epochs, losses_train, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Learning Curve: Training Loss')
    plt.grid(True)
    plt.legend()

    # Validation risk curve (right on the first row)
    plt.subplot(2, 2, 2)
    plt.plot(epochs, risks_val, label='Validation Risk', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Risk (MAE)')
    plt.title('Learning Curve: Validation Risk')
    plt.grid(True)
    plt.legend()

    # Convert performance data to DataFrame
    df_performance = pd.DataFrame(performance_data)

    # Group by lambda_decay and alpha and take the average to handle duplicates
    df_performance = df_performance.groupby(['lambda_decay', 'alpha'], as_index=False).agg({'validation_risk': 'mean'})

    # Pivot the DataFrame for heatmap (based on lambda_decay vs alpha for simplicity)
    heatmap_data = df_performance.pivot(index='lambda_decay', columns='alpha', values='validation_risk')

    # Heatmap plot at the bottom (spanning both columns)
    plt.subplot(2, 1, 2)
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title('Hyperparameter Heatmap: Validation Risk (MAE)')
    plt.xlabel('alpha')
    plt.ylabel(r"$\lambda_{decay}$")

    # Adjust layout to fit everything
    plt.tight_layout()
    plt.show()

# Plot the learning curves and hyperparameter heatmap for training loss and validation risk
plot_learning_curves(best_losses_train, best_risks_val, performance_data)



