import pandas as pd
import os
from sklearn.datasets import fetch_california_housing

def load_data(data_path="data/raw/house_prices.csv"):
    """
    Loads the dataset from the given path.
    If the dataset does not exist, it downloads the California Housing dataset
    and saves it as a CSV file to 'data_path' for local usage.
    """
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Downloading California Housing dataset...")
        # create data directory if it doesn't exist
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        california = fetch_california_housing()
        df = pd.DataFrame(california.data, columns=california.feature_names)
        df['Price'] = california.target # Target variable (Median House Value)
        
        df.to_csv(data_path, index=False)
        print(f"Dataset saved to {data_path}")
        return df
    
    print(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    return df

if __name__ == "__main__":
    df = load_data()
    print("Data loaded successfully. Shape:", df.shape)
    print(df.head())
