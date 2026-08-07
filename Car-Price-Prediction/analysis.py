import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

OUTPUT_DIR = 'outputs'
DATA_FILE = os.path.join('data', 'CarPrice_Assignment.csv')


def load_data():
    """Load the car price dataset and display initial dataset information."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    print('=' * 50)
    print('   CAR PRICE PREDICTION - ANALYSIS SCRIPT')
    print('=' * 50)
    print('DATA LOADING AND INITIAL EXPLORATION')
    print('=' * 50)

    print('Dataset shape:')
    print(df.shape)
    print('-' * 50)

    print('First 5 rows:')
    print(df.head())
    print('-' * 50)

    print('Last 5 rows:')
    print(df.tail())
    print('-' * 50)

    print('Column names:')
    print(list(df.columns))
    print('-' * 50)

    print('Data types:')
    print(df.dtypes)
    print('-' * 50)

    print('Null values per column:')
    print(df.isnull().sum())
    print('-' * 50)

    print('Total duplicate rows:')
    print(df.duplicated().sum())
    print('=' * 50)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the car price dataset and return the cleaned DataFrame."""
    print('\n' + '=' * 50)
    print('=== PHASE 2: DATA CLEANING ===')
    print('=' * 50)

    # Remove duplicate rows and report how many were removed.
    duplicate_count = df.duplicated().sum()
    print(f'Duplicate rows before cleaning: {duplicate_count}')
    df = df.drop_duplicates().reset_index(drop=True)
    print(f'Duplicate rows removed: {duplicate_count}')
    print('-' * 50)

    # Report null counts before cleaning.
    print('Null values before cleaning:')
    print(df.isnull().sum())
    print('-' * 50)

    # Drop rows where the target column price is missing.
    if 'price' in df.columns:
        price_nulls = df['price'].isnull().sum()
        print(f"price null rows to drop: {price_nulls}")
        df = df[df['price'].notna()].reset_index(drop=True)
    else:
        print('Warning: price column not found. No target rows dropped.')
    print('-' * 50)

    # Fill nulls using mode for categorical and median for numerical columns.
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == object or df[col].dtype.name == 'string':
                mode_value = df[col].mode(dropna=True)
                fill_value = mode_value.iloc[0] if not mode_value.empty else ''
                df[col] = df[col].fillna(fill_value)
                print(f"Filled nulls in categorical column '{col}' with mode: {fill_value}")
            else:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                print(f"Filled nulls in numerical column '{col}' with median: {median_value}")
    print('-' * 50)

    # Standardize string/categorical columns by trimming whitespace.
    string_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()

    # Ensure specific categorical columns have consistent Title Case values.
    categorical_cols = [
        'fueltype',
        'aspiration',
        'doornumber',
        'carbody',
        'drivewheel',
        'enginelocation',
        'enginetype',
        'cylindernumber',
        'fuelsystem',
    ]
    for col in categorical_cols:
        if col in df.columns:
            print(f"Unique values in '{col}' before standardization:")
            print(df[col].unique())
            df[col] = df[col].str.title()
            print(f"Unique values in '{col}' after standardization:")
            print(df[col].unique())
            print('-' * 50)
        else:
            print(f"Warning: '{col}' column not found in DataFrame.")
            print('-' * 50)

    # Remove outliers from price using the IQR method.
    if 'price' in df.columns:
        print('Removing price outliers using IQR method...')
        shape_before = df.shape
        q1 = df['price'].quantile(0.25)
        q3 = df['price'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)].shape[0]
        df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)].reset_index(drop=True)
        shape_after = df.shape
        print(f'Shape before outlier removal: {shape_before}')
        print(f'Shape after outlier removal: {shape_after}')
        print(f'Outlier rows removed: {outlier_count}')
    else:
        print('Warning: price column not found. Skipping outlier removal.')
    print('-' * 50)

    # Report null counts after cleaning to confirm there are no null values.
    print('Null values after cleaning:')
    print(df.isnull().sum())
    print('-' * 50)

    # Print final DataFrame info for the cleaned dataset.
    print('Final cleaned dataset info:')
    df.info()
    print('=' * 50)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features for car price prediction and return the updated DataFrame."""
    print('\n' + '=' * 50)
    print('=== PHASE 3: FEATURE ENGINEERING ===')
    print('=' * 50)

    # Calculate car age from the current year and the provided Year column.
    current_year = datetime.now().year
    if 'Year' in df.columns:
        df['Car_Age'] = current_year - df['Year']
        print('Sample Car_Age values:')
        print(df['Car_Age'].head())
        # Drop the original Year column once the new age feature is created.
        df = df.drop(columns=['Year'])
    else:
        # If the dataset does not include a Year column, the age feature cannot be computed.
        print("Warning: 'Year' column not found. Skipping Car_Age creation.")
    print('-' * 50)

    # Extract brand from car name and normalize text casing.
    if 'CarName' in df.columns:
        df['Brand'] = df['CarName'].astype(str).str.split().str[0].str.title()
        print('Brand distribution:')
        print(df['Brand'].value_counts())
        # Drop the original CarName column after extracting the brand.
        df = df.drop(columns=['CarName'])
    elif 'Car_Name' in df.columns:
        df['Brand'] = df['Car_Name'].astype(str).str.split().str[0].str.title()
        print('Brand distribution:')
        print(df['Brand'].value_counts())
        df = df.drop(columns=['Car_Name'])
    else:
        print("Warning: neither 'CarName' nor 'Car_Name' column found. Skipping Brand extraction.")
    print('-' * 50)

    # Report the updated DataFrame columns and a quick preview of the new features.
    print('Columns after feature engineering:')
    print(list(df.columns))
    print('-' * 50)

    print('DataFrame preview after feature engineering:')
    print(df.head())
    print('=' * 50)

    return df


def run_eda(df: pd.DataFrame) -> None:
    """Run exploratory data analysis and save charts to the outputs directory."""
    print('\n' + '=' * 50)
    print('=== PHASE 4: EDA & VISUALIZATIONS ===')
    print('=' * 50)

    # Create aliases for requested column names when the actual columns exist.
    if 'price' in df.columns and 'Selling_Price' not in df.columns:
        df['Selling_Price'] = df['price']
    if 'fueltype' in df.columns and 'Fuel_Type' not in df.columns:
        df['Fuel_Type'] = df['fueltype'].str.title()
    if 'Transmission' not in df.columns and 'transmission' in df.columns:
        df['Transmission'] = df['transmission'].str.title()

    # 1. Selling Price Distribution
    if 'Selling_Price' in df.columns:
        plt.figure()
        sns.histplot(df['Selling_Price'], kde=True, color='steelblue', bins=40)
        plt.title('Distribution of Selling Price')
        plt.xlabel('Selling Price (Lakhs)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, '01_price_distribution.png')
        plt.savefig(output_path)
        plt.close()
        skewness = df['Selling_Price'].skew()
        print(f'Saved Selling Price distribution to {output_path}')
        print(f'Selling Price skewness: {skewness:.4f}')
    else:
        print("Skipping price distribution: 'Selling_Price' column not available.")
    print('-' * 50)

    # 2. Price vs Fuel Type Box Plot
    if 'Selling_Price' in df.columns and 'Fuel_Type' in df.columns:
        plt.figure()
        sns.boxplot(data=df, x='Fuel_Type', y='Selling_Price', hue='Fuel_Type', palette='Set2', dodge=False)
        plt.title('Selling Price by Fuel Type')
        plt.xlabel('Fuel Type')
        plt.ylabel('Selling Price (Lakhs)')
        plt.legend([], [], frameon=False)
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, '02_price_vs_fuel.png')
        plt.savefig(output_path)
        plt.close()
        print(f'Saved price vs fuel type box plot to {output_path}')
    else:
        print("Skipping fuel type box plot: required columns not available.")
    print('-' * 50)

    # 3. Price vs Car Age Scatter Plot
    if 'Car_Age' in df.columns and 'Selling_Price' in df.columns:
        plt.figure()
        scatter_kwargs = {'x': 'Car_Age', 'y': 'Selling_Price', 'data': df}
        if 'Fuel_Type' in df.columns:
            scatter_kwargs['hue'] = 'Fuel_Type'
        sns.scatterplot(**scatter_kwargs)
        plt.title('Selling Price vs Car Age')
        plt.xlabel('Car Age (Years)')
        plt.ylabel('Selling Price (Lakhs)')
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, '03_price_vs_age.png')
        plt.savefig(output_path)
        plt.close()
        print(f'Saved price vs car age scatter plot to {output_path}')
    else:
        print("Skipping price vs car age scatter plot: required columns not available.")
    print('-' * 50)

    # 4. Top 10 Brands by Average Selling Price
    if 'Brand' in df.columns and 'Selling_Price' in df.columns:
        brand_mean = df.groupby('Brand')['Selling_Price'].mean().sort_values(ascending=False).head(10)
        plt.figure()
        plt.barh(brand_mean.index[::-1], brand_mean.values[::-1], color=plt.cm.viridis(np.linspace(0, 1, len(brand_mean))))
        plt.title('Top 10 Brands by Average Selling Price')
        plt.xlabel('Average Selling Price (Lakhs)')
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, '04_top10_brands.png')
        plt.savefig(output_path)
        plt.close()
        print(f'Saved top 10 brands chart to {output_path}')
    else:
        print("Skipping top brands chart: required columns not available.")
    print('-' * 50)

    # 5. Transmission vs Selling Price Box Plot
    if 'Selling_Price' in df.columns and 'Transmission' in df.columns:
        plt.figure()
        sns.boxplot(data=df, x='Transmission', y='Selling_Price', palette='Set1')
        plt.title('Selling Price by Transmission Type')
        plt.xlabel('Transmission')
        plt.ylabel('Selling Price (Lakhs)')
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, '05_price_vs_transmission.png')
        plt.savefig(output_path)
        plt.close()
        print(f'Saved transmission box plot to {output_path}')
    else:
        print("Skipping transmission box plot: 'Transmission' column not available.")
    print('=' * 50)


def encode_and_correlate(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables and generate a correlation heatmap."""
    print('\n' + '=' * 50)
    print('=== PHASE 5: ENCODING & CORRELATION ===')
    print('=' * 50)

    encoded_df = df.copy()

    # Create aliases for the expected categorical columns if the actual dataset uses different names.
    if 'price' in encoded_df.columns and 'Selling_Price' not in encoded_df.columns:
        encoded_df['Selling_Price'] = encoded_df['price']
    if 'fueltype' in encoded_df.columns and 'Fuel_Type' not in encoded_df.columns:
        encoded_df['Fuel_Type'] = encoded_df['fueltype'].str.title()
    if 'seller_type' in encoded_df.columns and 'Seller_Type' not in encoded_df.columns:
        encoded_df['Seller_Type'] = encoded_df['seller_type'].astype(str).str.title()
    if 'transmission' in encoded_df.columns and 'Transmission' not in encoded_df.columns:
        encoded_df['Transmission'] = encoded_df['transmission'].astype(str).str.title()

    categorical_cols = [
        col for col in ['Fuel_Type', 'Seller_Type', 'Transmission', 'Brand']
        if col in encoded_df.columns
    ]

    print('Categorical columns before encoding:')
    print(categorical_cols)
    print('-' * 50)

    shape_before = encoded_df.shape
    print(f'Shape before encoding: {shape_before}')

    if categorical_cols:
        encoded_df = pd.get_dummies(encoded_df, columns=categorical_cols, drop_first=True, dtype=int)
        shape_after = encoded_df.shape
        print(f'Shape after encoding: {shape_after}')
        print('New column names after encoding:')
        print(list(encoded_df.columns))
    else:
        print('No categorical columns found for one-hot encoding.')
        shape_after = shape_before
    print('-' * 50)

    # Compute correlation matrix on numeric features only.
    if 'Selling_Price' in encoded_df.columns:
        numeric_df = encoded_df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()

        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, '06_correlation_heatmap.png')
        plt.savefig(output_path)
        plt.close()
        print(f'Saved correlation heatmap to {output_path}')

        if 'Selling_Price' in corr_matrix.columns:
            price_corr = corr_matrix['Selling_Price'].drop('Selling_Price').abs().sort_values(ascending=False)
            print('Top 10 features most correlated with Selling_Price:')
            print(price_corr.head(10))
    else:
        print("Skipping correlation heatmap: 'Selling_Price' column not available.")
    print('=' * 50)

    return encoded_df


if __name__ == '__main__':
    df = load_data()
    cleaned_df = clean_data(df)
    engineered_df = engineer_features(cleaned_df)
    run_eda(engineered_df)
    encoded_df = encode_and_correlate(engineered_df)
