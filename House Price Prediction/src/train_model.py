import os
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
try:
    from src.data_loader import load_data
    from src.preprocessing import preprocess_data
except ModuleNotFoundError:
    from data_loader import load_data
    from preprocessing import preprocess_data

def train_multiple_models():
    print("=============================================")
    print("🚀 MULTI-MODEL BENCHMARK TRAINING STARTING")
    print("=============================================\n")
    
    # 1. Pipeline: Load and Preprocess advanced dataset
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # 2. Define our Models alongside their Hyperparameter Boundaries (Grid Search)
    models_to_tune = {
        'LinearRegression': (LinearRegression(), {}),
        'Ridge': (Ridge(), {'alpha': [0.1, 1.0, 10.0, 50.0, 100.0]}),
        'Lasso': (Lasso(max_iter=5000), {'alpha': [1.0, 10.0, 50.0, 100.0]}),
        'RandomForest': (RandomForestRegressor(random_state=42), {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        })
    }
    
    # Create directories for artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("\n[+] Scaler saved to models/scaler.pkl")
    
    # 3. Training & Tuning Loop
    trained_models = {}
    for name, (base_model, params) in models_to_tune.items():
        print(f"[⌛] Dynamically Tuning & Training: {name} (This may take a moment)...")
        
        if params:
            # Executes Hyperparameter Tuning searching for the absolute best parameters
            search = GridSearchCV(base_model, param_grid=params, cv=3, n_jobs=-1, scoring='r2')
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            print(f"     🔥 Ideal Hyperparameters Found: {search.best_params_}")
        else:
            best_model = base_model
            best_model.fit(X_train, y_train)
            print(f"     ✅ Successfully baseline trained: {name}")
            
        # Save optimal individual models
        joblib.dump(best_model, f"models/{name}_model.pkl")
        trained_models[name] = best_model
        
    print("\n🎉 ALL MODELS TRAINED SUCCESSFULLY!")
    print("Run `evaluate.py` to see the Leaderboard Benchmarks!")
    
    return trained_models

if __name__ == "__main__":
    train_multiple_models()
