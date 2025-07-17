import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker="AAPL", start="2015-01-01", end="2025-01-01"):
    stock = yf.download(ticker, start=start, end=end)
    stock.to_csv(f"{ticker}.csv") 
    print(f" Stock data for {ticker} saved as {ticker}.csv")

if __name__ == "__main__":
    fetch_stock_data()

