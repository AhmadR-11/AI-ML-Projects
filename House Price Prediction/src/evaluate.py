import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_loader import load_data
from preprocessing import preprocess_data

def evaluate(model_path="models/linear_regression_model.pkl", data_path="data/raw/train.csv"):
    print("Loading saved model and data for evaluation...")
    
    # Load Data
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Load Model
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}. Please run train_model.py first.")
        return
        
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    # Calculate Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("-" * 40)
    print("MODEL EVALUATION METRICS:")
    print("-" * 40)
    print(f"Mean Absolute Error (MAE):       {mae:.4f}")
    print(f"Mean Squared Error (MSE):        {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE):  {rmse:.4f}")
    print(f"R-squared Score (R2):            {r2:.4f}")
    print("-" * 40)
    print("-" * 40)
    
    # Visualizations
    print("Generating visualizations...")
    os.makedirs("visualizations", exist_ok=True)
    
    # 1. Actual vs Predicted Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Prices ($)", fontsize=12)
    plt.ylabel("Predicted Prices ($)", fontsize=12)
    plt.title("Actual vs. Predicted House Prices", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualizations/actual_vs_predicted.png")
    plt.show()  # Opens window for user to see
    
    # 2. Residuals (Errors) Plot
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, color='purple', bins=40)
    plt.axvline(x=0, color='red', linestyle='--', lw=2)
    plt.xlabel("Prediction Error (Actual - Predicted) [$]", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Errors (Residuals)", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualizations/residuals_distribution.png")
    plt.show()  # Opens window for user to see
    return {"mae": mae, "rmse": rmse, "r2": r2}

if __name__ == "__main__":
    evaluate()
