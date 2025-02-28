# model.py

# Import necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

def train_model(X_train, y_train):
    """
    Trains a Linear Regression model using the training data.

    Parameters:
    X_train (array-like): The feature matrix for training.
    y_train (array-like): The target values for training.

    Returns:
    model: A trained Linear Regression model.
    """
    # Initialize the Linear Regression model
    model = LinearRegression()

    # Train the model using the training data
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model's performance on the test data using multiple evaluation metrics.
    
    Parameters:
    model: The trained model to be evaluated.
    X_test (array-like): The feature matrix for testing.
    y_test (array-like): The true target values for testing.
    
    Returns:
    dict: A dictionary containing the evaluation metrics (MSE, MAE, RMSE, and R²).
    """
    # Predict the target values using the model on the test set
    y_pred = model.predict(X_test)
    
    # Calculate the Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)

    # Calculate the Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)

    # Calculate the Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    # Calculate the R² Score
    r2 = r2_score(y_test, y_pred)
    
    # Return a dictionary with the evaluation metrics
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2_Score': r2
    }

def plot_learning_curve(model, X_train, y_train):
    """
    Plots the learning curve for a given model to visualize how the model's performance improves
    with increasing training size.

    Parameters:
    model: The trained model.
    X_train (array-like): The feature matrix for training.
    y_train (array-like): The target values for training.
    """
    # Create the learning curve using cross-validation
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )

    # Calculate the mean and standard deviation of training and testing scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label="Training Score", color="blue", lw=2)
    plt.plot(train_sizes, test_mean, label="Cross-Validation Score", color="green", lw=2)

    # Plot the shaded areas for standard deviation
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="green", alpha=0.1)

    # Add labels and title
    plt.title("Learning Curve", fontsize=16)
    plt.xlabel("Training Size", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()