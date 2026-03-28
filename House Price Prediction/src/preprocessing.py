import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, target_column='SalePrice', test_size=0.2, random_state=42):
    """
    Cleans the data, splits into training and testing sets, and applies scaling.
    """
    print("Preprocessing data...")
    
    # Handle missing values (fill with median)
    if df.isnull().sum().any():
        print("Handling missing values...")
        df.fillna(df.median(numeric_only=True), inplace=True)
        
    # Separate Features (X) and Target (y)
    X = df.drop(columns=[target_column])
    X = X.select_dtypes(include=['number'])  # Keep only numeric columns for simplicity
    y = df[target_column]
    
    # Split into Train and Test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Feature Scaling (Standardization)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    print("Preprocessing completed.")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
