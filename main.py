import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score  # Added R²-Score
from src.data_preprocessing import preprocess_data
from src.utils import plot_feature_importance  # Function for Feature Importance Visualization

# --- Loading Data ---
# Assumption: The file "data/train.csv" is already cleaned and ready for processing.
df = pd.read_csv("data/train.csv")

# --- Data Preprocessing ---
# The preprocess_data function should return features and target variable (X, y)
X, y = preprocess_data(df)

# if X being a numpy ndarray, convert to DataFrame
X = pd.DataFrame(X, columns=df.drop(columns='SalePrice').columns)

# --- Data Partitioning ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Hyperparameter-Tuning with GridSearchCV ---
# Parameter grid for optimization
param_grid = {
    'n_estimators': [50, 100, 150],         # Number of trees
    'max_depth': [5, 10, 15, 20, 25, 30, None],  # Maximum depth of trees
    'min_samples_split': [2, 5, 10]         # Minimum samples to create a split
}

# Initialize the model
model = RandomForestRegressor(random_state=42)

# GridSearchCV
grid_search = GridSearchCV(
    estimator=model,                        # Model passed in
    param_grid=param_grid,                  # Hyperparameter grid
    cv=5,                                   # Number of cross-validation folds
    n_jobs=1,                               # Use all available processors
    verbose=1                               # Detailed output
)

# Train the GridSearch model
grid_search.fit(X_train, y_train)

# Output of best parameter
print("Best Parameter:", grid_search.best_params_)

# Train the model with the best parameters
best_model = grid_search.best_estimator_

# --- Evaluation ---
# Predictions on the test data
y_pred = best_model.predict(X_test)

# Calculate Mean Squared Error (MSE) & R²-Score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE) with best parameters: {mse}")
print(f"R² Score with best parameters: {r2}")

# --- Feature Importance Visualization ---
# Visualize feature importance
# Replace 'SalePrice' with the correct target column name if needed
plot_feature_importance(best_model, X.columns)