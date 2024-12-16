import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from imblearn.over_sampling import SMOTE  
from collections import Counter

from sklearn.metrics import confusion_matrix


# Working directory setup
dir_working = "/media/adebayo/Windows/UofACanada/study/codes/CMPUT566_ML/project/"
try:
    os.chdir(dir_working)
except FileNotFoundError:
    print(f"Error: Directory {dir_working} not found.")
    exit(1)

# Load dataset
users_data_wout_outliers = pd.read_pickle('online_history_cleaned_data_no_outliers_final_ml.pickle')
target_data = users_data_wout_outliers.churned
features = users_data_wout_outliers.drop(['Invoice', 'churned'], axis=1)

import numpy as np
from imblearn.over_sampling import SMOTE

def prepare_data(features, target, seed=42):
    """
    Prepares data by normalizing, splitting into train/val/test, and applying SMOTE oversampling,
    while adding a bias term (column of ones) to the features at the end.

    Args:
        features (pd.DataFrame): Input features.
        target (pd.Series): Target labels.
        seed (int): Random seed for reproducibility.
    
    Returns:
        dict: A dictionary containing processed train, validation, and test sets.
    """
    np.random.seed(seed)
    
    # Ensure features and target are numpy arrays for processing
    X = features.to_numpy()
    y = target.to_numpy()

    # Combine features and target for shuffling and splitting
    data = np.hstack((X, y.reshape(-1, 1)))
    
    # Shuffle the combined data
    np.random.shuffle(data)
    
    # Split features and target back
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    
    # Determine unique class proportions
    unique_classes, class_counts = np.unique(y, return_counts=True)
    
    # Indices for splits
    train_idx = int(0.7 * len(y))
    val_idx = int(0.85 * len(y))
    
    # Split data into 70% train, 15% validation, 15% test
    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)  # Shuffle indices within each class
        
        cls_train_idx = int(0.7 * len(cls_indices))
        cls_val_idx = int(0.85 * len(cls_indices))
        
        X_train.append(X[cls_indices[:cls_train_idx]])
        y_train.append(y[cls_indices[:cls_train_idx]])
        
        X_val.append(X[cls_indices[cls_train_idx:cls_val_idx]])
        y_val.append(y[cls_indices[cls_train_idx:cls_val_idx]])
        
        X_test.append(X[cls_indices[cls_val_idx:]])
        y_test.append(y[cls_indices[cls_val_idx:]])
    
    # Concatenate splits back together
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    X_val = np.vstack(X_val)
    y_val = np.concatenate(y_val)
    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)
    
    #print(f"Data before norm: {X_train[:2]}")
    # Normalize the data using training set statistics
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    
    #print(f"Data after norm: {X_train[:2]}")

    
    # Add a bias term (column of ones) to the features at the end
    X_train = np.concatenate([X_train, np.ones([X_train.shape[0], 1])], axis=1)  # Add bias term at the end
    X_val = np.concatenate([X_val, np.ones([X_val.shape[0], 1])], axis=1)  # Add bias term at the end
    X_test = np.concatenate([X_test, np.ones([X_test.shape[0], 1])], axis=1)  # Add bias term at the end
    
    #print(f"Data after bias: {X_train[:2]}")

    # Oversample the training data using SMOTE
    smote = SMOTE(random_state=seed)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"Shape dada: {X_train_smote.shape}")
    # Return processed data
    return {
        "X_train": X_train_smote,
        "y_train": y_train_smote,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }




# Sigmoid function
def sigmoid(z):
    """Sigmoid activation function for binary classification."""
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy_loss(y, t):
    """Binary cross-entropy loss for binary classification."""
    N = t.shape[0]
    return -np.mean(t * np.log(y + 1e-15) + (1 - t) * np.log(1 - y + 1e-15))

# Ridge Regression
import numpy as np


def ridge_regression(X_train, t_train, decay):
   
    # Number of features including the bias term
    n_features = X_train.shape[1]  # Already includes bias term
    
    # Identity matrix for regularization
    I = np.eye(n_features)  # Identity matrix of size (n_features x n_features)
    I[0, 0] = 0  # Do not regularize the intercept term

    # Closed-form solution using pseudo-inverse to avoid singular matrix errors
    w_ridge = np.linalg.pinv(X_train.T @ X_train + decay * I) @ X_train.T @ t_train
    
    # Calculate the Ridge regression loss (MSE + regularization term)
    predictions = np.dot(X_train, w_ridge)
    mse_loss = np.mean((predictions - t_train) ** 2)  # Mean Squared Error
    regularization_loss = decay * np.sum(w_ridge[1:] ** 2)  # Regularization term (skip the bias term)
    total_loss = mse_loss + regularization_loss

    return w_ridge, total_loss


def predict_linear_regression(X, w, t=None):

    y = np.dot(X, w)  # Linear prediction (no sigmoid)
    y_hat = (y >= 0.5).astype(int)
    
    # Calculate loss (Mean Squared Error) if true labels are provided
    loss = None
    if t is not None:
        loss = np.mean((y_hat - t) ** 2)  # Mean Squared Error loss

    return y_hat, loss


# Logistic Regression (Gradient Descent)
def train_logistic_regression(X, t, lr=0.1, epochs=1000):
    """Train logistic regression using gradient descent."""
    n, d = X.shape
    w = np.zeros(d)  # Initialize weights
    b = 0  # Initialize bias
    loss_history = []  # Store training loss history
    
    for _ in range(epochs):
        z = np.dot(X, w) + b
        y = sigmoid(z)
        error = y - t
        
        dw = np.dot(X.T, error) / n
        db = np.sum(error) / n
        
        w -= lr * dw
        b -= lr * db
        
        loss = binary_cross_entropy_loss(y, t)
        loss_history.append(loss)
    
    return w, b, loss_history

# Logistic Regression Prediction function
def predict_logistic_regression(X, w, b, t=None):
    """Generate predictions using logistic regression."""
    z = np.dot(X, w) + b
    y = sigmoid(z)
    t_hat = (y >= 0.5).astype(int)
    return t_hat

# SGD Prediction function
def predict_SGD(X, W, t=None):
    """Predict with SGD and evaluate loss and accuracy."""
    z = np.dot(X, W)
    y = sigmoid(z)
    t_hat = (y >= 0.5).astype(int)
    
    loss, acc = None, None
    if t is not None:
        loss = binary_cross_entropy_loss(y, t)
        acc = np.mean(t_hat == t) * 100

    return y, t_hat, loss, acc

# Train Logistic Regression using SGD
def train_SGD(X_train, y_train, X_val, t_val, alpha, batch_size, MaxEpoch, decay):
    """Train logistic regression using SGD with weight decay."""
    W = np.zeros(X_train.shape[1])  # Initialize weights
    train_losses, val_losses = [], []
    acc_best = 0
    epoch_best = 0

    for epoch in range(MaxEpoch):
        indices = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]
            
            y_pred, _, loss, _ = predict_SGD(X_batch, W, t=y_batch)
            grad_W = np.dot(X_batch.T, (y_pred - y_batch)) / batch_size + decay * W
            W -= alpha * grad_W
        
        _, _, train_loss, _ = predict_SGD(X_train, W, t=y_train)
        _, _, val_loss, val_acc = predict_SGD(X_val, W, t=t_val)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_acc > acc_best:
            acc_best = val_acc
            epoch_best = epoch
            W_best = W.copy()

    return epoch_best, acc_best, W_best, train_losses, val_losses


class MetricsCalculator:
    """Class for calculating precision, recall, accuracy, and f1-score."""
    
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        
    def accuracy(self):
        """Calculate accuracy."""
        return np.mean(self.y_true == self.y_pred)
    
    def precision(self):
        """Calculate precision."""
        tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        fp = np.sum((self.y_true == 0) & (self.y_pred == 1))
        return tp / (tp + fp) if (tp + fp) != 0 else 0
    
    def recall(self):
        """Calculate recall."""
        tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        fn = np.sum((self.y_true == 1) & (self.y_pred == 0))
        return tp / (tp + fn) if (tp + fn) != 0 else 0
    
    def f1_score(self):
        """Calculate F1 score."""
        precision = self.precision()
        recall = self.recall()
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0


def plot_confusion_matrix_with_percentages(cm, ax=None, tick_fontsize=14):
    """Plot confusion matrix with percentages below counts and text color depending on background intensity."""
    cm_percentage = cm / cm.sum(axis=1, keepdims=True) * 100  # Calculate percentages
    
    # Create a new axis if one isn't provided
    if ax is None:
        fig, ax = plt.subplots()

    # Create the heatmap with counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred: 0', 'Pred: 1'], 
                yticklabels=['True: 0', 'True: 1'], ax=ax, cbar=False, annot_kws={"size": 16})

    # Annotate percentages below the count values and change text color based on background intensity
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percentage = cm_percentage[i, j]
            
            # Get the background color of the current cell (the face color of the heatmap cell)
            cell_color = ax.collections[0].get_facecolor()[i * cm.shape[1] + j]
            
            # Convert RGB to grayscale to measure brightness of the background (luminosity)
            r, g, b, _ = cell_color
            brightness = 0.2126 * r + 0.7152 * g + 0.0722 * b  # Using the luminance formula
            
            # If the brightness is low (dark color), make the text white, else black
            text_color = 'white' if brightness < 0.5 else 'black'
            
            # Display the percentage below the count with the dynamic color
            ax.text(j+0.5, i + 0.25, f"({percentage:.2f}%)", ha='center', va='center', color=text_color, 
                    fontsize=14, fontweight='bold', backgroundcolor=cell_color)

    ax.set_title("Confusion Matrix with Percentages Below Counts")

    # Adjust tick font size for both x and y axes
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)

    plt.show()


import matplotlib.pyplot as plt

# Define a plotting function to show the learning curves
def plot_learning_curves(losses_train, risks_val, model_type='SGD', epochs=None):
    
    if model_type == 'SGD' and epochs is None:
        raise ValueError("SGD requires the number of epochs to be specified.")
    
    if model_type == 'Logistic' and epochs is None:
        epochs = range(1, len(losses_train) + 1)  # For Logistic, we use index as epoch-like progression
    
    # Plot the learning curves
    plt.figure(figsize=(10, 3))

    # Training loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses_train, label='Training Loss', color='blue')
    plt.xlabel('Epochs' if model_type == 'SGD' else 'Iterations')
    plt.ylabel('Training Loss')
    plt.title('Learning Curve: Training Loss')
    plt.grid(True)
    plt.legend()

    # Validation risk curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, risks_val, label='Validation Risk', color='orange')
    plt.xlabel('Epochs' if model_type == 'SGD' else 'Iterations')
    plt.ylabel('Validation Risk (MAE)')
    plt.title('Learning Curve: Validation Risk')
    plt.grid(True)
    plt.legend()

    # Show both plots
    plt.tight_layout()
    plt.show()



def hyperparameter_tuning(X_train, t_train, X_val, t_val, X_test, t_test):
    """Performs hyperparameter tuning and returns best model parameters."""
    alphas = [0.1, 0.01, 0.001]
    decays = [0.0, 0.0001, 0.001]
    MaxEpoch = 200
    MaxIter = 1 
    batch_size = 100
    all_metrics = []

    # Track the best validation accuracy for each model
    best_ridge_val_acc = 0
    best_logistic_val_acc = 0
    best_sgd_val_acc = 0

    # Track the best hyperparameters for each model
    best_ridge_params = None
    best_logistic_params = None
    best_sgd_params = None

    # Train and evaluate for each hyperparameter combination
    for alpha in alphas:
        for decay in decays:
            for epoch in range(MaxIter):
                # Train and evaluate Ridge Regression (decay is applied)
                print(f"Testing Ridge Regression with decay={decay}")
                w_ridge, loss_ridge = ridge_regression(X_train, t_train, decay)
                
                train_pred_ridge, _ = predict_linear_regression(X_train, w_ridge, t_train)  # No bias term
                val_pred_ridge, _ = predict_linear_regression(X_val, w_ridge, t_val)
                
                train_loss_ridge = binary_cross_entropy_loss(train_pred_ridge, t_train)
                val_acc_ridge = np.mean(val_pred_ridge == t_val) * 100
                print(f"train_loss_ridge: {train_loss_ridge}")

                all_metrics.append({
                    "model": "Ridge Regression",
                    "alpha": None,
                    "decay": decay,
                    "epoch": epoch,
                    "train_loss": train_loss_ridge,
                    "val_acc": val_acc_ridge,
                })

                # Keep track of the best combination based on validation accuracy
                if val_acc_ridge > best_ridge_val_acc:
                    best_ridge_val_acc = val_acc_ridge
                    best_ridge_params = {"decay": decay, "w_best": w_ridge}

                # Train and evaluate Logistic Regression (only alpha is used)
                print(f"Testing Logistic Regression with alpha={alpha}")
                w_logistic, b_logistic, loss_logistic = train_logistic_regression(X_train, t_train, lr=alpha, epochs=MaxEpoch)
                #print(f"loss_logistic: {loss_logistic}")
                train_pred_logistic = predict_logistic_regression(X_train, w_logistic, b_logistic, t_train)
                val_pred_logistic = predict_logistic_regression(X_val, w_logistic, b_logistic, t_val)
                
                train_loss_logistic = binary_cross_entropy_loss(train_pred_logistic, t_train)
                val_loss_logistic = binary_cross_entropy_loss(val_pred_logistic, t_val)
                val_acc_logistic = np.mean(val_pred_logistic == t_val) * 100
                print(f"train_loss_logistic: {train_loss_logistic}")
                
                all_metrics.append({
                    "model": "Logistic Regression",
                    "alpha": alpha,
                    "decay": None,
                    "epoch": epoch,
                    "train_loss": train_loss_logistic,
                    "val_acc": val_acc_logistic,
                })

                # Keep track of the best combination based on validation accuracy
                if val_acc_logistic > best_logistic_val_acc:
                    best_logistic_val_acc = val_acc_logistic
                    best_logistic_params = {"alpha": alpha, "w_best": w_logistic, "b_best": b_logistic}

                # Train and evaluate SGD (both alpha and decay are used)
                print(f"Testing SGD with alpha={alpha} and decay={decay}")
                epoch_best, val_acc_best, W_best, train_losses, valid_accs = train_SGD(
                    X_train, t_train, X_val, t_val, alpha, batch_size, MaxEpoch, decay
                )
                
                all_metrics.append({
                    "model": "SGD",
                    "alpha": alpha,
                    "decay": decay,
                    "epoch": epoch,
                    "train_loss": train_losses[-1],
                    "val_acc": val_acc_best,
                })

                # Keep track of the best combination based on validation accuracy
                if val_acc_best > best_sgd_val_acc:
                    best_sgd_val_acc = val_acc_best
                    best_sgd_params = {"alpha": alpha, "decay": decay, "w_best": W_best}

    # After all hyperparameter combinations are tested
    metrics_df = pd.DataFrame(all_metrics)

    # Summary metrics
    summary = metrics_df.groupby(["model", "alpha", "decay"]).agg(
        train_loss_mean=("train_loss", "mean"),
        val_acc_mean=("val_acc", "mean"),
    ).reset_index()

    print("\nSummary of Metrics:")
    print(summary)

    # Test the models with the best hyperparameters found during validation
    print("\nTesting with best hyperparameters:")
    
    # Create a list to store final metrics for each model
    final_metrics = []

    # Ridge Regression Test
    if best_ridge_params is not None:
        print(f"Testing Ridge Regression with best decay={best_ridge_params['decay']}")
        test_pred_ridge, _ = predict_linear_regression(X_test, best_ridge_params['w_best'])
        test_acc_ridge = np.mean(test_pred_ridge == t_test) * 100
        print(f"Test Accuracy for Ridge Regression: {test_acc_ridge}%")
        metrics_ridge = MetricsCalculator(t_test, test_pred_ridge)
        precision_ridge = metrics_ridge.precision()
        recall_ridge = metrics_ridge.recall()
        f1_ridge = metrics_ridge.f1_score()

        final_metrics.append({
            "model": "Ridge Regression",
            "accuracy": test_acc_ridge/100,
            "precision": precision_ridge,
            "recall": recall_ridge,
            "f1_score": f1_ridge,
        })

        # Confusion Matrix for Ridge
        cm_ridge = confusion_matrix(t_test, test_pred_ridge)
        plot_confusion_matrix_with_percentages(cm_ridge)

    # Logistic Regression Test
    if best_logistic_params is not None:
        print(f"Testing Logistic Regression with best alpha={best_logistic_params['alpha']}")
        test_pred_logistic = predict_logistic_regression(X_test, best_logistic_params['w_best'], best_logistic_params['b_best'])
        test_acc_logistic = np.mean(test_pred_logistic == t_test) * 100
        print(f"Test Accuracy for Logistic Regression: {test_acc_logistic}%")
        metrics_logistic = MetricsCalculator(t_test, test_pred_logistic)
        precision_logistic = metrics_logistic.precision()
        recall_logistic = metrics_logistic.recall()
        f1_logistic = metrics_logistic.f1_score()

        final_metrics.append({
            "model": "Logistic Regression",
            "accuracy": test_acc_logistic/100,
            "precision": precision_logistic,
            "recall": recall_logistic,
            "f1_score": f1_logistic,
        })

        # Confusion Matrix for Logistic
        cm_logistic = confusion_matrix(t_test, test_pred_logistic)
        plot_confusion_matrix_with_percentages(cm_logistic)
        #plot_learning_curves(train_loss_logistic, val_loss_logistic, model_type='Logistic')

    # SGD Test
    if best_sgd_params is not None:
        print(f"Testing SGD with best alpha={best_sgd_params['alpha']} and decay={best_sgd_params['decay']}")
        _, test_pred_sgd, _, _ = predict_SGD(X_test, best_sgd_params['w_best'], t_test)
        test_acc_sgd = np.mean(test_pred_sgd == t_test) * 100
        print(f"Test Accuracy for SGD: {test_acc_sgd}%")
        metrics_sgd = MetricsCalculator(t_test, test_pred_sgd)
        precision_sgd = metrics_sgd.precision()
        recall_sgd = metrics_sgd.recall()
        f1_sgd = metrics_sgd.f1_score()

        final_metrics.append({
            "model": "SGD",
            "accuracy": test_acc_sgd/100,
            "precision": precision_sgd,
            "recall": recall_sgd,
            "f1_score": f1_sgd,
        })

        # Confusion Matrix for SGD
        cm_sgd = confusion_matrix(t_test, test_pred_sgd)
        plot_confusion_matrix_with_percentages(cm_sgd)

        plot_learning_curves(train_losses, valid_accs, model_type='SGD', epochs=range(1, len(train_losses) + 1))

    # Create a DataFrame with final metrics
    final_metrics_df = pd.DataFrame(final_metrics)

    print("\nFinal Metrics (Test Set):")
    print(final_metrics_df)

    return best_ridge_params, best_logistic_params, best_sgd_params, summary, final_metrics_df



# Main function call
if __name__ == "__main__":
    data = prepare_data(features, target_data, seed=32)
    #metrics_df, summary, 
    best_ridge_params, best_logistic_params, best_sgd_params, summary, final_metrics_df = hyperparameter_tuning(
    data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test']
)

    print("best_ridge_params:")
    print(best_ridge_params)
    print("best_logistic_params:")
    print(best_logistic_params)
    print("best_sgd_params:")
    print(best_sgd_params)
    print("Metrics:")
    print(final_metrics_df)
   


    # Convert to long format
    df_long = final_metrics_df.melt(id_vars="model", var_name="Metric", value_name="Value")
    # Set a visually appealing color palette
    palette = sns.color_palette("Paired", n_colors=3)  # Options: 'Set1', 'Set2', 'Paired', 'Dark2', etc.

    # Plot
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df_long, x="Metric", y="Value", hue="model", palette=palette)

    # Customize
    plt.title("Model Comparison Across Metrics", fontsize=16)
    plt.xlabel("Metrics", fontsize=14)
    plt.ylabel("Values", fontsize=14)
    plt.legend(title="Model", fontsize=10)
    plt.tight_layout()

    # Show the plot
    plt.show()
    



