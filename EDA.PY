
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff


pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")


df = pd.read_csv("upi_transactions_2024.csv")  

print("\nDataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


df.replace([np.inf, -np.inf], np.nan, inplace=True)

threshold = len(df) * 0.5
df = df.loc[:, df.isnull().sum() < threshold]

for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include=['object']).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)


if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')


print("\nData Info:")
print(df.info())
print("\nData Description (numeric):")
print(df.describe())


plt.figure(figsize=(8, 5))
sns.histplot(df['amount (INR)'], bins=50, kde=True, color='blue')
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Amount (INR)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='transaction type', order=df['transaction type'].value_counts().index)
plt.title('Transaction Count by Type')
plt.xlabel('Transaction Type')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data=df, y='merchant_category', order=df['merchant_category'].value_counts().index)
plt.title('Transactions by Merchant Category')
plt.xlabel('Count')
plt.ylabel('Merchant Category')
plt.tight_layout()
plt.show()


if 'timestamp' in df.columns:
    df_ts = df[['timestamp', 'amount (INR)']].dropna()
    df_ts = df_ts.groupby('timestamp')['amount (INR)'].sum().reset_index()
    fig = px.line(df_ts, x='timestamp', y='amount (INR)',
                  title="Total Transaction Amount Over Time")
    fig.show()


df_numeric = df.select_dtypes(include=[np.number])
if not df_numeric.empty and df_numeric.shape[1] > 1:
    corr = df_numeric.corr()
    fig = ff.create_annotated_heatmap(
        z=corr.values.round(2),
        x=list(corr.columns),
        y=list(corr.columns),
        annotation_text=corr.round(2).astype(str).values,
        colorscale='Viridis',
        showscale=True
    )
    fig.update_layout(title_text="Interactive Correlation Heatmap", height=800)
    fig.show()


plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='fraud_flag')
plt.title('Fraud vs Non-Fraud Transactions')
plt.xlabel('Fraud Flag')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


if 'hour_of_day' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='hour_of_day', order=sorted(df['hour_of_day'].unique()))
    plt.title('Transactions by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


if 'day_of_week' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='day_of_week', order=sorted(df['day_of_week'].unique()))
    plt.title('Transactions by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

print("\nEDA Completed Successfully.")
