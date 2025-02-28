# src/hyperparameter_tuning.py

# Import necessary libraries
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint, uniform

def tune_hyperparameters(X_train, y_train):
    """
    Performs hyperparameter tuning for a RandomForestRegressor using RandomizedSearchCV.
    
    Parameters:
    X_train (array-like): The training feature matrix.
    y_train (array-like): The target vector for training.
        
    Returns:
    best_model: The best RandomForestRegressor model found after tuning.
    
    Hyperparameter search space:
    - n_estimators: Number of trees in the forest, randomly sampled between 50 and 200.
    - max_depth: Maximum depth of each tree, randomly sampled between 5 and 30.
    - min_samples_split: Minimum samples required to split an internal node, randomly sampled between 2 and 10.
    - min_samples_leaf: Minimum samples required at a leaf node, randomly sampled between 1 and 10.
    - max_features: Fraction of features to consider at each split, uniformly sampled between 0.1 and 1.0.
    - bootstrap: Boolean flag to use bootstrap samples.
    """
    
    # Define the hyperparameter search space
    param_dist = {
        'n_estimators': randint(50, 200),  # Number of trees to sample
        'max_depth': randint(5, 30),  # Depth of the trees
        'min_samples_split': randint(2, 10),  # Minimum samples to split a node
        'min_samples_leaf': randint(1, 10),  # Minimum samples at a leaf node
        'max_features': uniform(0.1, 0.9),  # Fraction of features to consider at each split
        'bootstrap': [True, False]  # Whether bootstrap sampling is used
    }
    
    # Initialize the base RandomForestRegressor model
    model = RandomForestRegressor(random_state=42)

    # Set up the RandomizedSearchCV with cross-validation (5-fold) and the defined hyperparameter space
    random_search = RandomizedSearchCV(
        estimator=model,  # The estimator to tune
        param_distributions=param_dist,  # The hyperparameter distribution to sample from
        n_iter=100,  # Number of random hyperparameter combinations to try
        cv=5,  # Number of folds in cross-validation
        verbose=1,  # Display detailed progress
        random_state=42,  # Set the random seed for reproducibility
        n_jobs=-1  # Use all available CPU cores to speed up the search
    )
    
    # Fit the RandomizedSearchCV on the training data to find the best hyperparameters
    random_search.fit(X_train, y_train)
    
    # Output the best hyperparameters found
    print("Best Parameters:", random_search.best_params_)
    
    # Return the best model found after hyperparameter tuning
    return random_search.best_estimator_