import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_data, preprocess_data
from src.hyperparameter_tuning import tune_hyperparameters
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
# We assume that 'data/train.csv' is either already cleaned or updated by the data updater.
df = load_data('data/train.csv')
# Print column names for debugging and to verify that the target column is present.
print("DataFrame columns:", df.columns.tolist())

# =============================================================================
# Step 3: Data Preprocessing
# =============================================================================
# Preprocess the dataset:
# - Handle missing values, encode categorical variables, create derived features, and scale numerical features.
X, y = preprocess_data(df)
# Convert X to a DataFrame (in case it's returned as a numpy array)
# using the original DataFrame's columns (excluding the target column 'SalePrice').
X = pd.DataFrame(X, columns=df.drop(columns='SalePrice').columns)

# =============================================================================
# Step 4: Data Partitioning
# =============================================================================
# Split the data into training and testing sets using an 80/20 split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================================================
# Step 5: Hyperparameter Tuning
# =============================================================================
# Instead of performing hyperparameter tuning inline,
# we now call a dedicated function from 'src/hyperparameter_tuning.py'
best_model = tune_hyperparameters(X_train, y_train)

# =============================================================================
# Step 6: Model Evaluation
# =============================================================================
# Evaluate the best model using multiple metrics:
# Mean Squared Error (MSE), Mean Absolute Error (MAE),
# Root Mean Squared Error (RMSE), and R² Score.
metrics = evaluate_model(best_model, X_test, y_test)
print("Evaluation Metrics:")
print(f"Mean Squared Error (MSE) with best parameters: {metrics['MSE']}")
print(f"Mean Absolute Error (MAE) with best parameters: {metrics['MAE']}")
print(f"Root Mean Squared Error (RMSE) with best parameters: {metrics['RMSE']}")
print(f"R² Score with best parameters: {metrics['R2_Score']}")

# =============================================================================
# Step 7: Visualize Feature Importance
# =============================================================================
# Visualize the feature importance using the preprocessed feature names.
plot_feature_importance(best_model, X.columns)