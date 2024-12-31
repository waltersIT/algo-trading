import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# List of major chip company stock tickers
chip_companies = ['PXLW', 'TRT', 'PRSO', 'LEDS', 'VLN', 'EMKR', 'SQNS', 'GSIT']

def fetch_financial_data(ticker):
    """Fetches historical stock data and financial metrics."""
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period='1y')
        if history.empty:
            return None

        # Add financial ratios (mock example, requires actual data)
        financials = {
            'P/E Ratio': stock.info.get('forwardPE', np.nan),
            'EPS': stock.info.get('trailingEps', np.nan),
            'Market Cap': stock.info.get('marketCap', np.nan)
        }

        # Add financial metrics to history DataFrame
        history['P/E Ratio'] = financials['P/E Ratio']
        history['EPS'] = financials['EPS']
        history['Market Cap'] = financials['Market Cap']

        return history
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def prepare_data(chip_companies):
    """Prepares dataset for machine learning."""
    all_data = []
    for ticker in chip_companies:
        data = fetch_financial_data(ticker)
        if data is not None and not data.empty:
            print(f"Successfully fetched data for {ticker}")
            data['Ticker'] = ticker
            all_data.append(data)
        else:
            print(f"Data for {ticker} is unavailable or empty.")
    
    if not all_data:
        raise ValueError("No valid data fetched for any tickers. Check tickers and data retrieval logic.")
    
    df = pd.concat(all_data, axis=0)
    df['Target'] = (df['Close'] < df['Open']).astype(int)  # Example target: 1 if price dropped
    df = df.dropna()
    return df


def train_model(df):
    """Trains a RandomForestClassifier to predict undervalued stocks."""
    features = ['P/E Ratio', 'EPS', 'Market Cap', 'Close', 'Open']
    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    return model

def plot_feature_importance(model, features):
    """Plots feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(features)), importances[indices], align="center")
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45)
    plt.show()

if __name__ == "__main__":
    data = prepare_data(chip_companies)
    print("Dataset Sample:\n", data.head())
    
    model = train_model(data)
    plot_feature_importance(model, ['P/E Ratio', 'EPS', 'Market Cap', 'Close', 'Open'])
