import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy import stats

def detect_and_cap_outliers(df, columns, threshold=3):
    """
    Detects outliers using Z-scores and caps them at the 3-sigma boundary
    to prevent them from heavily skewing the regression models.
    """
    df_clean = df.copy()
    for col in columns:
        # Calculate Z-scores (avoiding NaNs temporarily)
        z_scores = np.abs(stats.zscore(df_clean[col].fillna(df_clean[col].median())))
        
        # Cap outliers at the upper and lower bounds if Z > 3
        upper_bound = df_clean[col].mean() + threshold * df_clean[col].std()
        lower_bound = df_clean[col].mean() - threshold * df_clean[col].std()
        
        df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
        df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
    return df_clean

def preprocess_data(df, target_column='SalePrice', test_size=0.2, random_state=42):
    print("--- 1. FEATURE ENGINEERING ---")
    df_engineered = df.copy()
    
    # Feature 1: Total House Age (When it was sold minus when it was built/remodeled)
    if 'YrSold' in df.columns and 'YearRemodAdd' in df.columns:
        df_engineered['TotalHouseAge'] = df_engineered['YrSold'] - df_engineered['YearRemodAdd']
        print("[+] Created New Feature: 'TotalHouseAge'")
        
    # Feature 2: Total Square Footage (Above ground + Basement)
    if 'GrLivArea' in df.columns and 'TotalBsmtSF' in df.columns:
        df_engineered['TotalSquareFootage'] = df_engineered['GrLivArea'] + df_engineered['TotalBsmtSF'].fillna(0)
        print("[+] Created New Feature: 'TotalSquareFootage'")
        
    # Feature 3: Total Porch Area (Combining all exterior lounging areas)
    porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    if all(c in df.columns for c in porch_cols):
        df_engineered['TotalPorchArea'] = df_engineered[porch_cols].sum(axis=1)
        print("[+] Created New Feature: 'TotalPorchArea'")

    # Drop target & ID
    if target_column not in df_engineered.columns:
        raise ValueError(f"Target variable {target_column} Not Found!")
        
    X = df_engineered.drop(columns=[target_column, 'Id'], errors='ignore')
    y = df_engineered[target_column]
    
    # Split Numeric and Categorical Columns
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=['number']).columns.tolist()
    
    print("\n--- 2. MISSING VALUES & OUTLIERS ---")
    # Advanced Missing Value Imputation
    print("[+] Imputing numeric missing values with Median...")
    num_imputer = SimpleImputer(strategy='median')
    X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
    
    print("[+] Imputing categorical missing values with Most Frequent Strategy...")
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
    
    # Z-Score Outlier Treatment
    print("[+] Applying Z-Score Outlier Capping (Sigma=3) on Numerics...")
    X = detect_and_cap_outliers(X, numeric_cols)
    
    print("\n--- 3. CATEGORICAL ENCODING ---")
    # Select a few ordinal variables for Label Encoding (meaning order matters, e.g., Quality)
    ordinal_candidates = ['ExterQual', 'KitchenQual', 'BsmtQual', 'HeatingQC']
    label_encode_cols = [c for c in categorical_cols if c in ordinal_candidates]
    
    print(f"[+] Applying Label Encoding to Ordinal Features: {label_encode_cols}")
    le = LabelEncoder()
    for col in label_encode_cols:
        X[col] = le.fit_transform(X[col].astype(str))
        
    # Apply One-Hot Encoding to the rest (nominal values where order doesn't matter)
    onehot_cols = [c for c in categorical_cols if c not in label_encode_cols]
    print(f"[+] Applying One-Hot Encoding tracking to {len(onehot_cols)} nominal features...")
    # Using Pandas get_dummies for rapid sparse matrix expansion mapping
    X = pd.get_dummies(X, columns=onehot_cols, drop_first=True)
    
    # Drop any remaining unnamable columns to ensure scaler doesn't crash
    X = X.select_dtypes(include=['number'])

    # 4. Final Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    print("\n--- 4. FINAL SCALING ---")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    print("\n✅ Preprocessing Pipeline Finished Completely!")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    try:
        from src.data_loader import load_data
    except ModuleNotFoundError:
        from data_loader import load_data
    df = load_data()
    X_t, X_te, y_t, y_te, scl = preprocess_data(df)
    print(f"Final Data Shape: Train={X_t.shape}, Test={X_te.shape}")
