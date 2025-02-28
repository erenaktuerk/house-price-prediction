from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def train_model(X_train, y_train):
    """
    Trains a Linear Regression model using the provided training data.
    
    Parameters:
        X_train (array-like): Training features.
        y_train (array-like): Target values.
    
    Returns:
        model (LinearRegression): Trained Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model using multiple metrics including MSE, MAE, RMSE, and R².
    
    This function calculates:
      - Mean Squared Error (MSE): The average squared difference between predictions and actual values.
      - Mean Absolute Error (MAE): The average absolute difference between predictions and actual values.
      - Root Mean Squared Error (RMSE): The square root of MSE, representing error in the same units as the target.
      - R² Score: The proportion of variance explained by the model.
      
    These metrics provide a comprehensive view of model performance.
    
    Parameters:
        model: Trained regression model.
        X_test (array-like): Test features.
        y_test (array-like): Actual target values.
    
    Returns:
        metrics (dict): A dictionary containing the calculated evaluation metrics.
    """
    # Generate predictions for the test data
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Return all metrics in a dictionary
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2_Score": r2}

def save_model(model, file_path):
    """
    Saves the trained model to the specified file path using joblib.
    
    Parameters:
        model: Trained model to be saved.
        file_path (str): Path to save the model.
    """
    joblib.dump(model, file_path)

def load_model(file_path):
    """
    Loads a saved model from the specified file path.
    
    Parameters:
        file_path (str): Path to the saved model.
    
    Returns:
        model: Loaded model.
    """
    return joblib.load(file_path)

def plot_feature_importance(model, feature_names):
    """
    Plots the feature importance for a model (supports both Linear Regression and tree-based models).
    
    For linear models, the feature importance is derived from the model coefficients.
    For tree-based models, it uses the attribute 'feature_importances_'.
    
    Parameters:
        model: Trained model (e.g., LinearRegression or RandomForestRegressor).
        feature_names: List of feature names (should be a 1-dimensional array-like).
    
    Raises:
        AttributeError: If the model does not have either coefficients or feature importances.
    """
    # Ensure feature_names is a list (if a DataFrame is passed, extract column names)
    if isinstance(feature_names, pd.DataFrame):
        feature_names = feature_names.columns.tolist()
    
    try:
        # For linear models, derive feature importance from coefficients
        feature_importance = pd.Series(model.coef_, index=feature_names).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        feature_importance.plot(kind='bar')
        plt.title("Feature Importance (Linear Model)")
        plt.xlabel("Features")
        plt.ylabel("Coefficient Value")
        plt.show()
    except AttributeError:
        # For tree-based models, use the feature_importances_ attribute
        if hasattr(model, "feature_importances_"):
            feature_importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
            plt.figure(figsize=(10, 6))
            feature_importance.plot(kind='bar')
            plt.title("Feature Importance (Tree-Based Model)")
            plt.xlabel("Features")
            plt.ylabel("Importance")
            plt.show()
        else:
            # If neither attribute is present, raise an error
            raise AttributeError(
                "Model does not have coefficients or feature importances. Ensure the model is a valid regression model."
            )