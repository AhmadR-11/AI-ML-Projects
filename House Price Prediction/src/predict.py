import joblib
import pandas as pd
from data_loader import load_data

def test_single_predictions():
    print("Loading saved model and scaler...")
    
    try:
        model = joblib.load("models/linear_regression_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
    except FileNotFoundError:
        print("Error: Model or Scaler not found. Please run train_model.py first.")
        return

    # Load data and pick 5 random houses to test on
    df = load_data("data/raw/train.csv")
    
    print("Selecting 5 random houses to test the model...")
    # Grab 5 random rows from the dataset
    random_houses = df.sample(n=5, random_state=101)
    
    # Get the actual real-world prices to compare against
    actual_prices = random_houses['SalePrice'].values
    
    # Drop the price column (since that's what we are trying to predict)
    # and keep only numeric columns (exactly as we did in training)
    X_new = random_houses.drop(columns=['SalePrice'])
    X_new = X_new.select_dtypes(include=['number'])
    
    # Handle any missing values
    X_new.fillna(X_new.median(numeric_only=True), inplace=True)
    
    # Scale the features using our saved Scaler
    X_new_scaled = scaler.transform(X_new)
    
    # Make our predictions!
    predictions = model.predict(X_new_scaled)
    
    print("\n" + "="*50)
    print("LIVE TESTING RESULTS")
    print("="*50)
    
    for i in range(5):
        actual = actual_prices[i]
        predicted = predictions[i]
        difference = actual - predicted
        
        print(f"House {i+1}:")
        print(f"  Actual Price:    ${actual:,.2f}")
        print(f"  Predicted Price: ${predicted:,.2f}")
        
        if difference < 0:
            print(f"  Error: Model OVERESTIMATED by ${abs(difference):,.2f}")
        else:
            print(f"  Error: Model UNDERESTIMATED by ${abs(difference):,.2f}")
        print("-" * 50)

if __name__ == "__main__":
    test_single_predictions()
