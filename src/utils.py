from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import pandas as pd

def train_model(X_train, y_train):
    """
    Trains a Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model using various metrics.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"MSE": mse, "MAE": mae, "R2_Score": r2}

def save_model(model, file_path):
    """
    Saves the trained model to a file.
    """
    joblib.dump(model, file_path)

def load_model(file_path):
    """
    Loads a saved model from a file.
    """
    return joblib.load(file_path)

def plot_feature_importance(model, feature_names):
    """
    Plots the feature importance for a model (Linear Regression or Random Forest).

    Parameters:
    - model: Trained model (LinearRegression or RandomForest)
    - feature_names: List of feature names
    """
    # convert feature_names into a list of column names, if necessary
    if isinstance(feature_names, pd.DataFrame):
        feature_names = feature_names.columns.tolist()

    try:
        # for linear models
        feature_importance = pd.Series(model.coef_, index=feature_names).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        feature_importance.plot(kind='bar')
        plt.title("Feature Importance (Linear Model)")
        plt.xlabel("Features")
        plt.ylabel("Coefficient Value")
        plt.show()
    except AttributeError:
        # for models like randomforest
        if hasattr(model, "feature_importances_"):
            feature_importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
            plt.figure(figsize=(10, 6))
            feature_importance.plot(kind='bar')
            plt.title("Feature Importance (Tree-Based Model)")
            plt.xlabel("Features")
            plt.ylabel("Importance")
            plt.show()
        else:
            raise AttributeError(
                "Model does not have coefficients or feature importances. Ensure the model is a valid regression model."
            )
    # plotting of the feature importance
    plt.figure(figsize=(10, 6))
    feature_importance.plot(kind='bar')
    plt.title("Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance Value")
    plt.show()