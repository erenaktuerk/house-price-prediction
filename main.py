import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from src.data_preprocessing import load_data, preprocess_data
from src.utils import plot_feature_importance, evaluate_model

# =============================================================================
# Step 1: Data Update Step
# =============================================================================
# Run the data updater script to fetch and update real-time data.
# This ensures that the latest data is available before processing.
subprocess.run(["python", "src/data_updater.py"])

# =============================================================================
# Step 2: Load the Data
# =============================================================================
# Load the training data from CSV.
# We assume that 'data/train.csv' is already cleaned or has been updated by the data updater.
df = load_data('data/train.csv')
# Print column names for debugging and to verify that the target column is present.
print("DataFrame columns:", df.columns.tolist())

# =============================================================================
# Step 3: Data Preprocessing
# =============================================================================
# Preprocess the dataset:
# - Handle missing values
# - Encode categorical variables
# - Create derived features (e.g., TotalArea, HouseAge, etc.)
# - Scale numerical features
X, y = preprocess_data(df)

# Convert X to a DataFrame in case it is returned as a numpy array.
# We use the original DataFrame's columns, excluding the target column 'SalePrice'.
X = pd.DataFrame(X, columns=df.drop(columns='SalePrice').columns)

# =============================================================================
# Step 4: Data Partitioning
# =============================================================================
# Split the data into training and testing sets using an 80/20 split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================================================
# Step 5: Hyperparameter Tuning with GridSearchCV
# =============================================================================
# Define the parameter grid for the RandomForestRegressor.
param_grid = {
    'n_estimators': [50, 100, 150],           # Number of trees in the forest
    'max_depth': [5, 10, 15, 20, 25, 30, None], # Maximum depth of each tree
    'min_samples_split': [2, 5, 10]             # Minimum samples required to split an internal node
}

# Initialize the RandomForestRegressor with a fixed random state for reproducibility.
model = RandomForestRegressor(random_state=42)

# Set up GridSearchCV to perform exhaustive hyperparameter tuning with 5-fold cross-validation.
grid_search = GridSearchCV(
    estimator=model,        # The base model to be tuned
    param_grid=param_grid,  # The grid of hyperparameters to search over
    cv=5,                   # Number of cross-validation folds
    n_jobs=1,               # Number of parallel jobs (adjust as needed)
    verbose=1               # Detailed output during the search
)

# Train the model using GridSearchCV on the training data.
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found by GridSearchCV.
print("Best Parameter:", grid_search.best_params_)

# Retrieve the best estimator (model) from GridSearchCV.
best_model = grid_search.best_estimator_

# =============================================================================
# Step 6: Model Evaluation
# =============================================================================
# Evaluate the model using multiple metrics: Mean Squared Error (MSE),
# Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.
# The evaluate_model function returns these metrics in a dictionary.
metrics = evaluate_model(best_model, X_test, y_test)

# Print out all evaluation metrics.
print("Evaluation Metrics:")
print(f"Mean Squared Error (MSE) with best parameters: {metrics['MSE']}")
print(f"Mean Absolute Error (MAE) with best parameters: {metrics['MAE']}")
print(f"Root Mean Squared Error (RMSE) with best parameters: {metrics['RMSE']}")
print(f"R² Score with best parameters: {metrics['R2_Score']}")

# =============================================================================
# Step 7: Visualize Feature Importance
# =============================================================================
# Visualize the importance of each feature as determined by the model.
# We pass the column names from the preprocessed feature DataFrame.
plot_feature_importance(best_model, X.columns)