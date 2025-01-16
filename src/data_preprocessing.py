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
    - Encodes categorical variables using LabelEncoder.
    - Creates new features.
    - Scales numerical features.
    """
    # Fill missing values with the median
    df.fillna(df.median(numeric_only=True), inplace=True)    
    # Create a new feature 'TotalArea'
    # Combines living area and total number of rooms
    if "GrLivArea" in df.columns and "TotRmsAbvGrd" in df.columns:
        df["TotalArea"] = df["GrLivArea"] + df["TotRmsAbvGrd"]
    
    # Drop irrelevant columns (like 'Id')
    if "Id" in df.columns:
        df.drop(columns=["Id"], inplace=True)
    
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Apply LabelEncoder to categorical columns
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])
    
    # Separate features (X) and target (y)
    X = df.drop(columns=["SalePrice"])  # Assuming "SalePrice" is the target
    y = df["SalePrice"]
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y