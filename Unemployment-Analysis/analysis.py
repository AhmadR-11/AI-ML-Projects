import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

CSV_PATH = 'data/Unemployment in India.csv'
OUTPUT_DIR = 'outputs'


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the unemployment data from a CSV file and print initial diagnostics."""
    df = pd.read_csv(csv_path)
    print('=== DATA SHAPE ===')
    print(df.shape)
    print('\n=== DATA PREVIEW ===')
    print(df.head())
    print('\n=== DATA TYPES ===')
    print(df.dtypes)
    print('\n=== MISSING VALUES ===')
    print(df.isnull().sum())
    print('\n=== COLUMN LIST ===')
    print(df.columns.tolist())
    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str, str]:
    """Clean and prepare the unemployment DataFrame for later analysis."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    date_column = None
    for col in df.columns:
        if 'date' in col.lower():
            date_column = col
            break

    if date_column is not None:
        df[date_column] = pd.to_datetime(df[date_column], dayfirst=True, errors='coerce')

    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    if df.shape[0] < initial_rows:
        print(f"\n=== DROPPED DUPLICATES ===\nRemoved {initial_rows - df.shape[0]} duplicate rows.")

    if date_column is not None:
        df = df.dropna(subset=[date_column])
        df['Month'] = df[date_column].dt.month
        df['Year'] = df[date_column].dt.year

    print('\n=== CLEANED DATA INFO ===')
    print(df.info())
    print('\n=== CLEANED DATA DESCRIPTION ===')
    print(df.describe(include='all'))

    unemployment_col = None
    for col in df.columns:
        if 'unemployment' in col.lower() and 'rate' in col.lower():
            unemployment_col = col
            break
    if unemployment_col is None and 'Estimated Unemployment Rate (%)' in df.columns:
        unemployment_col = 'Estimated Unemployment Rate (%)'

    region_col = None
    for col in df.columns:
        if col.lower() in {'region', 'state', 'area'}:
            region_col = col
            break

    return df, date_column, region_col, unemployment_col


def run_eda(df: pd.DataFrame, region_col: str, unemployment_col: str) -> None:
    """Run exploratory data analysis on the cleaned unemployment dataset."""
    if region_col is not None and unemployment_col is not None:
        print('\n=== REGION-WISE AVERAGE UNEMPLOYMENT RATE ===')
        region_avg = df.groupby(region_col)[unemployment_col].mean().sort_values(ascending=False)
        print(region_avg)
    else:
        print('\n=== REGION-WISE AVERAGE UNEMPLOYMENT RATE ===')
        print('Region or unemployment rate column not detected.')

    if 'Month' in df.columns and unemployment_col is not None:
        print('\n=== MONTH-WISE AVERAGE UNEMPLOYMENT RATE ===')
        month_avg = df.groupby('Month')[unemployment_col].mean().sort_index()
        print(month_avg)
    else:
        print('\n=== MONTH-WISE AVERAGE UNEMPLOYMENT RATE ===')
        print('Month column or unemployment rate column not detected.')

    if 'Year' in df.columns and unemployment_col is not None:
        print('\n=== YEAR-WISE AVERAGE UNEMPLOYMENT RATE ===')
        year_avg = df.groupby('Year')[unemployment_col].mean().sort_index()
        print(year_avg)
    else:
        print('\n=== YEAR-WISE AVERAGE UNEMPLOYMENT RATE ===')
        print('Year column or unemployment rate column not detected.')


def plot_timeseries(df: pd.DataFrame, date_column: str, region_col: str, unemployment_col: str) -> None:
    """Plot a time-series line chart for the top 3 regions by average unemployment rate."""
    if date_column is None or region_col is None or unemployment_col is None:
        print('\n=== TIME-SERIES CHART ===')
        print('Required region, unemployment rate, or date columns not detected.')
        return

    region_avg = df.groupby(region_col)[unemployment_col].mean().sort_values(ascending=False)
    top_regions = region_avg.head(3).index.tolist()
    df_top = df[df[region_col].isin(top_regions)].sort_values(by=[date_column, region_col])

    plt.figure(figsize=(12, 6))
    markers = ['o', 's', '^']
    for i, region in enumerate(top_regions):
        region_data = df_top[df_top[region_col] == region]
        plt.plot(region_data[date_column], region_data[unemployment_col],
                 marker=markers[i % len(markers)], label=region)

    plt.title('Unemployment Rate Over Time - Top 3 Regions')
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'timeseries_chart.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nChart saved: {output_path}")


def plot_top10(df: pd.DataFrame, region_col: str, unemployment_col: str) -> None:
    """Plot a horizontal bar chart of the top 10 regions by average unemployment rate."""
    if region_col is None or unemployment_col is None:
        print('\n=== TOP 10 BAR CHART ===')
        print('Required region or unemployment rate column not detected.')
        return

    top10_regions = df.groupby(region_col)[unemployment_col].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette('viridis', len(top10_regions))
    y_positions = range(len(top10_regions))
    plt.barh(y_positions, top10_regions.values, color=palette)
    plt.yticks(y_positions, top10_regions.index)

    for index, value in enumerate(top10_regions.values):
        plt.text(value + 0.1, index, f'{value:.2f}', va='center')

    plt.title('Top 10 Regions by Average Unemployment Rate')
    plt.xlabel('Average Unemployment Rate (%)')
    plt.ylabel('Region')
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'top10_bar_chart.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nBar chart saved: {output_path}")


def plot_heatmap(df: pd.DataFrame) -> None:
    """Plot a correlation heatmap for unemployment, employment, and participation metrics."""
    numeric_cols = [
        'Estimated Unemployment Rate (%)',
        'Estimated Employed',
        'Estimated Labour Participation Rate (%)'
    ]
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    if len(numeric_cols) != 3:
        print('\n=== CORRELATION HEATMAP ===')
        print('Required numeric columns not found in the dataset:', numeric_cols)
        return

    corr_matrix = df[numeric_cols].corr()
    print('\n=== CORRELATION MATRIX ===')
    print(corr_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation: Unemployment, Employment & Labour Participation')
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nCorrelation heatmap saved: {output_path}")


def plot_covid_comparison(df: pd.DataFrame, date_column: str, unemployment_col: str) -> None:
    """Plot a Pre-COVID vs Post-COVID average unemployment rate comparison."""
    if date_column is None or unemployment_col is None:
        print('\n=== COVID COMPARISON ===')
        print('Required date or unemployment rate column not detected.')
        return

    pre_covid = df[df[date_column] < '2020-03-01']
    post_covid = df[df[date_column] >= '2020-03-01']

    pre_mean = pre_covid[unemployment_col].mean()
    post_mean = post_covid[unemployment_col].mean()

    print('\n=== PRE-COVID VS POST-COVID AVERAGE UNEMPLOYMENT RATE ===')
    print(f'Pre-COVID average unemployment rate: {pre_mean:.2f}%')
    print(f'Post-COVID average unemployment rate: {post_mean:.2f}%')

    pct_diff = ((post_mean - pre_mean) / pre_mean * 100) if pre_mean != 0 else float('nan')

    plt.figure(figsize=(8, 6))
    periods = ['Pre-COVID', 'Post-COVID']
    values = [pre_mean, post_mean]
    colors = ['green', 'red']
    bars = plt.bar(periods, values, color=colors)

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f'{value:.2f}%', ha='center', va='bottom')

    diff_text = f'Change: {pct_diff:+.2f}%'
    plt.text(0.5, max(values) * 0.95, diff_text, ha='center', va='top', fontsize=11, color='black')

    plt.title('Pre-COVID vs Post-COVID Average Unemployment Rate')
    plt.ylabel('Average Unemployment Rate (%)')
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'covid_comparison.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nCOVID comparison chart saved: {output_path}")


def main() -> None:
    """Execute the full unemployment analysis workflow."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('=== Unemployment Analysis - Pakistan ===')

    df = load_data(CSV_PATH)
    df, date_column, region_col, unemployment_col = clean_data(df)
    run_eda(df, region_col, unemployment_col)
    plot_timeseries(df, date_column, region_col, unemployment_col)
    plot_top10(df, region_col, unemployment_col)
    plot_heatmap(df)
    plot_covid_comparison(df, date_column, unemployment_col)

    print('=== Analysis Complete! Check outputs/ folder ===')


if __name__ == '__main__':
    main()
