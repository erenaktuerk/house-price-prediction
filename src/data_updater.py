import requests
import pandas as pd
import os

def fetch_real_time_data():
    """
    Fetch real-time data from an external API.
    
    This function demonstrates how to retrieve live data.
    In a production scenario, you would replace the simulation below
    with a real API call using requests.get() and process the returned JSON.
    
    Example for a real API call:
    
        url = "https://api.example.com/real_estate_data"
        headers = {
            "x-api-key": "YOUR_API_KEY",
            "x-api-host": "api.example.com"
        }
        response = requests.get(url, headers=headers)
        data = response.json()  # Assume JSON structure: {'results': [...]}
        df_new = pd.DataFrame(data["results"])
    
    For demonstration purposes, we simulate the data:
    """
    # Simulated data (replace with actual API call in production)
    data = {
        'MSSubClass': [20, 60],
        'MSZoning': ['RL', 'RM'],
        'LotFrontage': [80, 70],
        'LotArea': [9600, 11250],
        # ... add additional features as required ...
        'SalePrice': [200000, 250000],
        'TotalArea': [2500, 3000]
    }
    df_new = pd.DataFrame(data)
    return df_new

def update_csv(file_path, new_data):
    """
    Update the CSV file with the new data.
    
    If the CSV file already exists, append the new data to it.
    If it doesn't exist, create a new CSV file with the new data.
    
    Parameters:
        file_path (str): The full path to the CSV file.
        new_data (pd.DataFrame): The new data to be added.
    """
    if os.path.exists(file_path):
        # Load existing data from the CSV
        existing_data = pd.read_csv(file_path)
        # Append new data
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        updated_data = new_data
    
    # Save the updated data back to the CSV file without the index
    updated_data.to_csv(file_path, index=False)
    print(f"Updated {file_path} with {len(new_data)} new records.")

if __name__ == "__main__":
    # Fetch real-time data from the external API (simulated)
    real_time_data = fetch_real_time_data()
    
    # Define the path to the data directory and the CSV file (e.g., train.csv)
    data_dir = os.path.join(os.getcwd(), "data")
    train_file = os.path.join(data_dir, "train.csv")
    
    # Update the CSV file with the newly fetched data
    update_csv(train_file, real_time_data)