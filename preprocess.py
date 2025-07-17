import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file="AAPL.csv"):
    # Read the CSV, skipping extra header rows
    df = pd.read_csv(file, skiprows=2)  # ⬅️ Skipping first two lines

    # Rename columns (since original headers are missing)
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

    # Ensure 'Date' and 'Close' columns exist
    if "Date" not in df.columns or "Close" not in df.columns:
        raise KeyError(" 'Date' or 'Close' column missing! Check CSV format.")

    # Convert 'Date' column to datetime format
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    # Normalize Close prices
    scaler = MinMaxScaler()
    df["Close"] = scaler.fit_transform(df[["Close"]])

    # Save processed data
    df.to_csv("processed_stock_data.csv")
    print(" Data Preprocessing Done! Saved as 'processed_stock_data.csv'")

    return df, scaler

if __name__ == "__main__":
    preprocess_data()

