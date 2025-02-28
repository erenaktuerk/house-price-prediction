import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    """
    Loads the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocesses the dataset:
    - Handles missing values by replacing them with the median.
    - Creates new derived features (TotalArea, HouseAge, RemodelAge, QualityRatio).
    - Drops irrelevant columns (like 'Id').
    - Encodes categorical variables using LabelEncoder.
    - Separates features and target.
    - Scales numerical features.
    
    New Derived Features:
    - TotalArea: Sum of 'GrLivArea' and 'TotRmsAbvGrd' (if available).
    - HouseAge: Difference between 'YrSold' and 'YearBuilt' (if both available).
    - RemodelAge: Difference between 'YrSold' and 'YearRemodAdd' (if both available).
    - QualityRatio: Ratio of 'OverallQual' to 'OverallCond' (if both available).
    """
    # Fill missing values with the median (numeric columns only)
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    # Create new derived features
    
    # TotalArea: Combination of living area and number of rooms
    if "GrLivArea" in df.columns and "TotRmsAbvGrd" in df.columns:
        df["TotalArea"] = df["GrLivArea"] + df["TotRmsAbvGrd"]
    
    # HouseAge: Age of the house at time of sale
    if "YrSold" in df.columns and "YearBuilt" in df.columns:
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    
    # RemodelAge: Time since last remodeling at time of sale
    if "YrSold" in df.columns and "YearRemodAdd" in df.columns:
        df["RemodelAge"] = df["YrSold"] - df["YearRemodAdd"]
    
    # QualityRatio: Ratio of overall quality to overall condition
    if "OverallQual" in df.columns and "OverallCond" in df.columns:
        # Vermeide Division durch 0; falls OverallCond 0 ist, setze den Wert auf NaN
        df["QualityRatio"] = df["OverallQual"] / df["OverallCond"].replace(0, pd.NA)
    
    # Drop irrelevant columns, e.g. 'Id'
    if "Id" in df.columns:
        df.drop(columns=["Id"], inplace=True)
    
    # Identify categorical columns for encoding
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Apply LabelEncoder to categorical columns
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])
    
    # Separate features (X) and target (y)
    # Assuming 'SalePrice' is the target variable
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]
    
    # Scale numerical features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y