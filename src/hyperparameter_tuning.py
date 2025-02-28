# src/hyperparameter_tuning.py

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint, uniform

def tune_hyperparameters(X_train, y_train):
    """
    Performs hyperparameter tuning for RandomForestRegressor using RandomizedSearchCV.
    
    Parameters:
        X_train (array-like): Training feature matrix.
        y_train (array-like): Target vector for training.
        
    Returns:
        best_model: The best RandomForestRegressor model found after tuning.
    
    The hyperparameter search space is defined as follows:
      - n_estimators: Number of trees in the forest, randomly sampled between 50 and 200.
      - max_depth: Maximum depth of each tree, randomly sampled between 5 and 30.
      - min_samples_split: Minimum samples required to split an internal node, randomly sampled between 2 and 10.
      - min_samples_leaf: Minimum samples required at a leaf node, randomly sampled between 1 and 10.
      - max_features: Fraction of features to consider at each split, uniformly sampled between 0.1 and 1.0.
      - bootstrap: Boolean flag to use bootstrap samples.
    """
    # Define the hyperparameter distributions
    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': randint(5, 30),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 10),
        'max_features': uniform(0.1, 0.9),  # uniform(a, b) samples in [a, a+b)
        'bootstrap': [True, False]
    }
    
    # Initialize the base model
    model = RandomForestRegressor(random_state=42)
    
    # Set up RandomizedSearchCV with 5-fold cross-validation
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=100,        # Number of random combinations to sample
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1          # Use all available processors
    )
    
    # Fit RandomizedSearchCV on the training data
    random_search.fit(X_train, y_train)
    
    print("Best Parameters:", random_search.best_params_)
    
    # Return the best estimator found
    return random_search.best_estimator_