"""
Car Price Prediction - Data cleaning and model training helpers
=======================================
This module exposes reusable functions for loading, cleaning, encoding,
and training car price prediction models. It is intentionally free of
matplotlib/seaborn so a Dash app can import it cleanly.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_FILE = os.path.join('data', 'car_data.csv')
ALT_DATA_FILE = os.path.join('data', 'CarPrice_Assignment.csv')


def _resolve_data_path():
    """Resolve the dataset path, preferring data/car_data.csv."""
    if os.path.exists(DATA_FILE):
        return DATA_FILE
    if os.path.exists(ALT_DATA_FILE):
        return ALT_DATA_FILE
    raise FileNotFoundError(
        f"Dataset not found. Expected '{DATA_FILE}' or '{ALT_DATA_FILE}'."
    )


def load_and_clean_data() -> pd.DataFrame:
    """Load the car dataset and return a cleaned DataFrame."""
    data_path = _resolve_data_path()
    df = pd.read_csv(data_path)

    # Normalize target and categorical aliases.
    if 'price' in df.columns:
        df['Selling_Price'] = df['price']
    if 'fueltype' in df.columns:
        df['Fuel_Type'] = df['fueltype']
    if 'seller_type' in df.columns:
        df['Seller_Type'] = df['seller_type']
    if 'transmission' in df.columns:
        df['Transmission'] = df['transmission']
    if 'CarName' in df.columns:
        df['Car_Name'] = df['CarName']

    # Remove duplicate rows.
    df = df.drop_duplicates().reset_index(drop=True)

    # Drop rows missing the target.
    if 'Selling_Price' in df.columns:
        df = df[df['Selling_Price'].notna()].reset_index(drop=True)

    # Fill remaining missing values.
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == object or df[col].dtype.name == 'string':
                mode_values = df[col].mode(dropna=True)
                fill_value = mode_values.iloc[0] if not mode_values.empty else ''
                df[col] = df[col].fillna(fill_value)
            else:
                df[col] = df[col].fillna(df[col].median())

    # Standardize categorical values to Title Case.
    for col in ['Fuel_Type', 'Seller_Type', 'Transmission']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # Extract Brand from the first word of Car_Name.
    if 'Car_Name' in df.columns:
        df['Brand'] = df['Car_Name'].astype(str).str.split().str[0].str.title()
        df = df.drop(columns=['Car_Name'])
    elif 'CarName' in df.columns:
        df['Brand'] = df['CarName'].astype(str).str.split().str[0].str.title()
        df = df.drop(columns=['CarName'])

    # Create Car_Age from Year and drop Year.
    if 'Year' in df.columns:
        df['Car_Age'] = datetime.now().year - pd.to_numeric(df['Year'], errors='coerce')
        df = df.drop(columns=['Year'])

    # Remove outliers from Selling_Price using the IQR method.
    if 'Selling_Price' in df.columns:
        q1 = df['Selling_Price'].quantile(0.25)
        q3 = df['Selling_Price'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df['Selling_Price'] >= lower_bound) & (df['Selling_Price'] <= upper_bound)].reset_index(drop=True)

    # Drop raw alias columns that should not be used after normalization.
    alias_columns = ['price', 'fueltype', 'seller_type', 'transmission']
    df = df.drop(columns=[col for col in alias_columns if col in df.columns])

    return df


def get_encoded_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a one-hot encoded DataFrame for the selected categorical columns."""
    encoded_df = df.copy()
    categorical_cols = [
        col for col in ['Fuel_Type', 'Seller_Type', 'Transmission', 'Brand']
        if col in encoded_df.columns
    ]
    if categorical_cols:
        encoded_df = pd.get_dummies(encoded_df, columns=categorical_cols, drop_first=True, dtype=int)
    return encoded_df


def train_all_models(df_encoded: pd.DataFrame):
    """Train regression models and return models, test data, results, and feature names."""
    if 'Selling_Price' not in df_encoded.columns:
        raise ValueError("The encoded data must include a 'Selling_Price' column.")

    X = df_encoded.drop(columns=['Selling_Price'])
    y = df_encoded['Selling_Price']

    X = X.select_dtypes(include=[np.number])
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        results.append({'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'R2_Score': r2})

    results_df = pd.DataFrame(results).sort_values(by='R2_Score', ascending=False).reset_index(drop=True)
    return models, X_test, y_test, results_df, feature_names


if __name__ == '__main__':
    df = load_and_clean_data()
    df_enc = get_encoded_data(df)
    models, X_test, y_test, results_df, feature_names = train_all_models(df_enc)
    print(results_df)
