
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def fetch_stock_data(ticker: str, interval: str = "1h", period: str = "1y"):
    stock = yf.Ticker(ticker)
    df = stock.history(interval=interval, period=period)
    return df.dropna()

def calculate_macd(df):
    short_ema = df["Close"].ewm(span=12, adjust=False).mean()
    long_ema = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_Line"] = short_ema - long_ema
    df["Signal_Line"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()
    return df

def calculate_rsi(df, window=14):
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def calculate_adx(df, window=14):
    high_diff = df["High"].diff()
    low_diff = df["Low"].diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    tr = pd.concat([
        df["High"] - df["Low"],
        abs(df["High"] - df["Close"].shift()),
        abs(df["Low"] - df["Close"].shift())
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(window=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
    df["ADX"] = ((abs(plus_di - minus_di) / (plus_di + minus_di)) * 100).rolling(window=window).mean()
    return df

def calculate_moving_averages(df, window_sma=50, window_ema=20):
    df["SMA"] = df["Close"].rolling(window=window_sma).mean()
    df["EMA"] = df["Close"].ewm(span=window_ema, adjust=False).mean()
    return df

import pandas as pd
import numpy as np

def calculate_di(df, period=14):
    """Calculate +DI, -DI, and ADX."""
    df["High-Low"] = df["High"] - df["Low"]
    df["High-PrevClose"] = np.abs(df["High"] - df["Close"].shift(1))
    df["Low-PrevClose"] = np.abs(df["Low"] - df["Close"].shift(1))
    
    df["+DM"] = np.where((df["High"] - df["High"].shift(1)) > (df["Low"].shift(1) - df["Low"]), 
                         np.maximum(df["High"] - df["High"].shift(1), 0), 0)
    df["-DM"] = np.where((df["Low"].shift(1) - df["Low"]) > (df["High"] - df["High"].shift(1)), 
                         np.maximum(df["Low"].shift(1) - df["Low"], 0), 0)
    
    df["TR"] = df[["High-Low", "High-PrevClose", "Low-PrevClose"]].max(axis=1)
    
    df["+DM_Smooth"] = df["+DM"].rolling(period).sum()
    df["-DM_Smooth"] = df["-DM"].rolling(period).sum()
    df["TR_Smooth"] = df["TR"].rolling(period).sum()
    
    df["+DI"] = 100 * (df["+DM_Smooth"] / df["TR_Smooth"])
    df["-DI"] = 100 * (df["-DM_Smooth"] / df["TR_Smooth"])
    
    df["DX"] = 100 * (np.abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"]))
    df["ADX"] = df["DX"].rolling(period).mean()

    return df

def calculate_indicators(df):
    df = calculate_macd(df)
    df = calculate_rsi(df)
    df = calculate_adx(df)
    df = calculate_moving_averages(df)
    # Assuming df has 'High', 'Low', 'Close' columns
    df = calculate_di(df) 
    return df


def identify_trade_signals(df):
    # Add necessary technical indicators (pre-calculated in DataFrame)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
    df['MACD_Histogram'] = df['MACD_Line'] - df['Signal_Line']
    
    # Entry conditions for LONG positions
    long_entries = df[
        (df["Close"] < df["SMA"]) &  # Price above short-term SMA # Bullish SMA crossover
        (df["MACD_Line"] > df["Signal_Line"]) & 
        (df["ADX"] >= 20) & (df["ADX"] <= 50) &# MACD bullish crossover
        (df["+DI"] > df["-DI"]) &
        (df["RSI"].between(50, 70)) 
         # Strong but not overbought

    ]
    
    # Entry conditions for SHORT positions
    short_entries = df[
        (df["Close"] > df["SMA"]) &  # Price below short-term SMA  # Bearish SMA crossover
        (df["MACD_Line"] < df["Signal_Line"]) &  
        (df["ADX"] >= 20) & (df["ADX"] < 50) &
        (df["RSI"].between(30, 45)) &
        (df["-DI"] > df["+DI"]) 
        # Weak but not oversold
    ]
    
    return long_entries, short_entries

def plot_trade_signals(df, long_entries, short_entries, ticker):
    plt.figure(figsize=(12, 6))
    
    # Plot Closing Prices
    plt.plot(df.index, df["Close"], label="Close Price", color='blue', linewidth=1)
    plt.plot(df.index, df["SMA"], label="50-day SMA", color='orange', linestyle='dashed')

    # Plot Long Entries (Green Dots)
    plt.scatter(long_entries.index, long_entries["Close"], label="Long Entry", marker="o", color="green", s=50)

    # Plot Short Entries (Red Dots)
    plt.scatter(short_entries.index, short_entries["Close"], label="Short Entry", marker="o", color="red", s=50)

    plt.title(f"{ticker} Stock Price & Entry Positions")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()

def main(ticker: str, interval: str = "1h", period: str = "1y"):
    df = fetch_stock_data(ticker, interval, period)
    df = calculate_indicators(df)
    long_entries, short_entries = identify_trade_signals(df)
    
    print("Long Entries:\n", long_entries.tail())
    print("\nShort Entries:\n", short_entries.tail())

    # Plot the results
    plot_trade_signals(df, long_entries, short_entries, ticker)

    return long_entries, short_entries

if __name__ == "__main__":
    stock_ticker = "TCS.NS"
    main(stock_ticker)
