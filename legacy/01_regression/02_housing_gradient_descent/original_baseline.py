#!/usr/bin/env python3

import pickle as pickle
import numpy as np

import os

dir_working = "/media/adebayo/Windows/UofACanada/study/codes/CMPUT566_ML/"

os.chdir(dir_working)


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


def train(X_train, y_train, X_val, y_val):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    # initialization
    w = np.zeros([X_train.shape[1], 1])
    # w: (d+1)x1

    losses_train = []
    risks_val = []

    w_best = None
    risk_best = 10000
    epoch_best = 0

    for epoch in range(MaxIter):

        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train/batch_size))):

            X_batch = X_train[b*batch_size: (b+1)*batch_size]
            y_batch = y_train[b*batch_size: (b+1)*batch_size]

            y_hat_batch, loss_batch, _ = predict(X_batch, w, y_batch)
            loss_this_epoch += loss_batch

            # TODO: Your code here
            # Mini-batch gradient descent
            gradient = (2 / batch_size) * X_batch.T @ (y_hat_batch - y_batch) + 2 * decay * w
            w -= alpha * gradient  # Update weights using gradient descent

        # TODO: Your code here
        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        avg_loss_train = loss_this_epoch / N_train
        losses_train.append(avg_loss_train)
        # 2. Perform validation on the validation set by the risk
        _, _, risk_val = predict(X_val, w, y_val.reshape(-1, 1))
        risks_val.append(risk_val)

        # 3. Keep track of the best validation epoch, risk, and the weights
        if risk_val < risk_best:
            risk_best = risk_val
            w_best = w.copy()
            epoch_best = epoch

    # Return some variables as needed
     # Print epoch stats
        print(f"Epoch {epoch}: Training Loss = {avg_loss_train}, Validation Risk = {risk_val}")

    print(f"Best validation risk {risk_best} at epoch {epoch_best}")

    return w_best, losses_train, risks_val
    


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

#####################
# setting

alpha = 0.001      # learning rate
batch_size = 10    # batch size
MaxIter = 100        # Maximum iteration
decay = 0.0          # weight decay


# TODO: Your code here
w_best, losses_train, risks_val = train(X_train, y_train, X_val, y_val)


# Perform test by the weights yielding the best validation performance

# Report numbers and draw plots as required.
y_hat_test, _, risk_test = predict(X_test, w_best, y_test.reshape(-1, 1))
print(f"Test risk (absolute error): {risk_test}")

import matplotlib.pyplot as plt

# Define a plotting function to show the learning curves
def plot_learning_curves(losses_train, risks_val):
    epochs = range(1, len(losses_train) + 1)
    
    # Plot the training loss
    plt.figure(figsize=(10, 3))

    # Training loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses_train, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Learning Curve: Training Loss')
    plt.grid(True)
    plt.legend()

    # Validation risk curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, risks_val, label='Validation Risk', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Risk (MAE)')
    plt.title('Learning Curve: Validation Risk')
    plt.grid(True)
    plt.legend()

    # Show both plots
    plt.tight_layout()
    plt.show()

# Plot the learning curves for training loss and validation risk
plot_learning_curves(losses_train, risks_val)