import os
import joblib
from sklearn.linear_model import LinearRegression
from data_loader import load_data
from preprocessing import preprocess_data

def train(model_path="models/linear_regression_model.pkl"):
    print("Starting training process...")
    
    # Load and preprocess
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Initialize Model
    print("Initializing Linear Regression...")
    model = LinearRegression()
    
    # Train
    print("Training model (this might take a few seconds)...")
    model.fit(X_train, y_train)
    
    # Save Model and Scaler
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, "models/scaler.pkl")
    print(f"Model successfully saved to {model_path}")
    print("Scaler successfully saved to models/scaler.pkl")
    
    return model

if __name__ == "__main__":
    train()
    print("Training script finished.")
