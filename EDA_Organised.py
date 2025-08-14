import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import os
import re

def sanitize_filename(tag):
    """Convert tag names to valid filenames"""
    return re.sub(r'[\\/*?:"<>|()]', '_', tag)

def create_dirs():
    dirs = {
        'time_series': 'eda_results/time_series',
        'distributions': 'eda_results/distributions',
        'correlation': 'eda_results/correlation'
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    threshold = len(df) * 0.5
    cols_to_keep = [col for col in df.columns if col == 'timestamp' or df[col].isnull().sum() < threshold]
    df = df[cols_to_keep]
    for col in df.select_dtypes(include=np.number):
        df[col].fillna(df[col].median(), inplace=True)
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        except Exception as e:
            print(f"Timestamp error: {e}")
    return df
    
def save_time_series(df, dirs):
    """Save time series plots for all numeric columns as HTML using Plotly Express"""
    
    # Ensure 'timestamp' exists and is valid
    if 'timestamp' not in df.columns:
        print("Timestamp column missing, skipping time series plots.")
        return
    
    # Convert to datetime if not already
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])  # Drop rows with invalid timestamps
    if df.empty:
        print("All timestamps are NaT after parsing. Skipping time series.")
        return

    # Sort by timestamp
    df = df.sort_values('timestamp')

    # Select numeric columns (excluding 'timestamp')
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if col == 'timestamp':
            continue
        
        try:
            # Drop rows where this column is NaN (but timestamp is already valid)
            df_ts = df[['timestamp', col]].dropna()
            if df_ts.empty:
                print(f"No valid data for {col}, skipping.")
                continue

            fig = px.line(
                df_ts, x='timestamp', y=col,
                title=f"Time Series of {col}",
                labels={'timestamp': 'Timestamp', col: 'Value'}
            )
            fig.update_traces(line=dict(width=3.5, color='#ff0000'))
            fig.update_layout(
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='black'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='black'),
                plot_bgcolor='blue'
            )

            safe_name = sanitize_filename(col)
            file_path = os.path.join(dirs['time_series'], f'ts_{safe_name}.html')
            fig.write_html(file_path)
        
        except Exception as e:
            print(f"Failed time series for {col}: {str(e)}")

def save_distributions(df, dirs):
    numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col != 'timestamp']
    for col in numeric_cols:
        try:
            data = df[col].dropna()
            if data.empty:
                print(f"No valid data for distribution plot of {col}, skipping.")
                continue
            plt.figure(figsize=(10, 6))
            sns.histplot(data, bins=30, kde=True, color='blue', alpha=0.5)
            plt.title(f"Distribution: {col}", fontsize=14)
            plt.xlabel(col, fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.grid(True, alpha=0.3)
            safe_name = sanitize_filename(col)
            plt.savefig(f"{dirs['distributions']}/dist_{safe_name}.png", bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Failed distribution for {col}: {str(e)}")

def save_correlation(df, dirs):
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        print("Not enough numeric columns for correlation matrix.")
        return
    try:
        plt.figure(figsize=(18, 12))
        cg = sns.clustermap(numeric_df.corr(), linewidths=0.5, cmap='coolwarm', center=0)
        plt.savefig(f"{dirs['correlation']}/clustered_corr.png", bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Clustered heatmap failed: {str(e)}")
    try:
        corr = numeric_df.corr().round(2)
        fig = ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.columns),
            annotation_text=corr.astype(str).values,
            colorscale='thermal',
            showscale=True
        )
        fig.update_layout(title_text="Interactive Correlation Matrix", height=1000)
        fig.write_html(f"{dirs['correlation']}/interactive_corr.html")
    except Exception as e:
        print(f"Interactive heatmap failed: {str(e)}")

if __name__ == "__main__":
    dirs = create_dirs()
    df = load_data("upi_transactions_2024.csv")
    save_time_series(df, dirs)
    save_distributions(df, dirs)
    save_correlation(df, dirs)
    print("EDA completed successfully. Check the 'eda_results' folder.")
