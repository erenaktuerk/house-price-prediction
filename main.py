import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score  # Added R²-Score
from src.data_preprocessing import preprocess_data
from src.utils import plot_feature_importance  # Function for Feature Importance Visualization
from src.utils import evaluate_model

# --- Data Update Step ---
# Run the data updater script to fetch and update real-time data
# This ensures that the latest data is available before processing.
subprocess.run(["python", "src/data_updater.py"])

# --- Loading Data ---
# Assumption: The file "data/train.csv" is already cleaned and ready for processing,
# or has been updated by the data_updater script.
df = pd.read_csv("data/train.csv")

# Print column names to verify the target column (for debugging purposes)
print("DataFrame columns:", df.columns.tolist())

# --- Data Preprocessing ---
# The preprocess_data function should return features (X) and target variable (y)
X, y = preprocess_data(df)

# Convert X to a DataFrame in case it is a numpy.ndarray,
# using the original DataFrame's columns (excluding the target column 'SalePrice').
X = pd.DataFrame(X, columns=df.drop(columns='SalePrice').columns)

# --- Data Partitioning ---
# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Hyperparameter-Tuning with GridSearchCV ---
# Define the parameter grid for optimization.
param_grid = {
    'n_estimators': [50, 100, 150],         # Number of trees
    'max_depth': [5, 10, 15, 20, 25, 30, None],  # Maximum depth of trees
    'min_samples_split': [2, 5, 10]         # Minimum samples to create a split
}

# Initialize the Random Forest Regressor model with a fixed random state for reproducibility.
model = RandomForestRegressor(random_state=42)

# Set up GridSearchCV to perform exhaustive hyperparameter tuning with 5-fold cross-validation.
grid_search = GridSearchCV(
    estimator=model,                        # Model to be tuned
    param_grid=param_grid,                  # Hyperparameter grid
    cv=5,                                   # Number of cross-validation folds
    n_jobs=1,                               # Run in a single process (adjust as needed)
    verbose=1                               # Display detailed output during the search
)

# Train the model using GridSearchCV on the training data.
grid_search.fit(X_train, y_train)

# Output the best hyperparameters found by GridSearchCV.
print("Best Parameter:", grid_search.best_params_)

# Retrieve the best estimator (model) from GridSearchCV.
best_model = grid_search.best_estimator_

# --- Evaluation ---
# Generate predictions on the test data using the best model
y_pred = best_model.predict(X_test)

# Evaluate the model using the enhanced evaluation function
metrics = evaluate_model(best_model, X_test, y_test)

# Print out all evaluation metrics
print(f"Mean Squared Error (MSE) with best parameters: {metrics['MSE']}")
print(f"Mean Absolute Error (MAE) with best parameters: {metrics['MAE']}")
print(f"Root Mean Squared Error (RMSE) with best parameters: {metrics['RMSE']}")
print(f"R² Score with best parameters: {metrics['R2_Score']}")

# --- Feature Importance Visualization ---
# Visualize the importance of each feature.
# Here we pass X.columns (the column names from the preprocessed feature DataFrame)
plot_feature_importance(best_model, X.columns)