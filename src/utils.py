# utils.py

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

def plot_feature_importance(model, feature_names):
    """
    Plots the feature importance for a given model. Assumes the model has a feature_importances_ attribute.

    Parameters:
    model: The trained model.
    feature_names: List of feature names to match the importance values.
    """
    try:
        # Check if model has the feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            raise AttributeError("Model does not have 'feature_importances_' attribute")
        
        # Get feature importance from the model
        feature_importances = model.feature_importances_

        # Create a bar plot of feature importances
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, feature_importances, color='green')
        plt.xlabel('Feature Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.title('Feature Importance Plot', fontsize=16)
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Error in plot_feature_importance: {e}")

def plot_learning_curve(model, X_train, y_train):
    """
    Plots the learning curve for a given model to visualize how the model's performance improves
    with increasing training size.

    Parameters:
    model: The trained model.
    X_train (array-like): The feature matrix for training.
    y_train (array-like): The target values for training.
    """
    try:
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

    except Exception as e:
        print(f"Error in plot_learning_curve: {e}")

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the performance of a given model using multiple metrics.

    Parameters:
    model: The trained model.
    X_test (array-like): The feature matrix for testing.
    y_test (array-like): The target values for testing.
    
    Returns:
    dict: A dictionary with model evaluation metrics (MSE, MAE, RMSE, R² Score).
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    try:
        # Make predictions using the model
        y_pred = model.predict(X_test)

        # Calculate the evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Return the metrics in a dictionary
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2_Score': r2
        }

        return metrics

    except Exception as e:
        print(f"Error in evaluate_model: {e}")
        return None