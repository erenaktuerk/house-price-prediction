import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_data, preprocess_data
from src.hyperparameter_tuning import tune_hyperparameters
from src.utils import plot_feature_importance, evaluate_model, plot_learning_curve
from sklearn.metrics import mean_squared_error

# =============================================================================
# Step 1: Data Update Step
# =============================================================================
# Run the data updater script to fetch and update real-time data.
subprocess.run(["python", "src/data_updater.py"])

# =============================================================================
# Step 2: Load the Data
# =============================================================================
df = load_data('data/train.csv')
print("DataFrame columns:", df.columns.tolist())

# =============================================================================
# Step 3: Data Preprocessing
# =============================================================================
X, y = preprocess_data(df)
X = pd.DataFrame(X, columns=df.drop(columns='SalePrice').columns)

# =============================================================================
# Step 4: Data Partitioning
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================================================
# Step 5: Hyperparameter Tuning
# =============================================================================
# Now we call the hyperparameter tuning function to find the best model
best_model = tune_hyperparameters(X_train, y_train)

# =============================================================================
# Step 6: Model Evaluation
# =============================================================================
# Evaluate the best model using various metrics
metrics = evaluate_model(best_model, X_test, y_test)
print("Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {metrics['MSE']}")
print(f"Mean Absolute Error (MAE): {metrics['MAE']}")
print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']}")
print(f"R² Score: {metrics['R2_Score']}")

# =============================================================================
# Step 7: Visualize Feature Importance
# =============================================================================
# Visualize the feature importance using the preprocessed feature names
plot_feature_importance(best_model, X.columns)

# =============================================================================
# Step 8: Visualize Learning Curves (Optional)
# =============================================================================
# Optionally, we can plot learning curves to assess overfitting/underfitting
plot_learning_curve(best_model, X_train, y_train)