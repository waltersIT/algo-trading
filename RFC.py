import finance_data
import pandas as pd
import numpy as np
import newsTrading as nt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class RFC:
    def __init__(self, chip_companies):
        self.chip_companies = chip_companies
        self.finance_data = finance_data.FinanceData()

    def prepare_data(self):
        """Fetches and prepares dataset with additional features."""
        all_data = []

        for ticker in self.chip_companies:
            data = self.finance_data.fetch_financial_data(ticker)
            if data is not None and not data.empty:
                print(f"Successfully fetched data for {ticker}")
                data['Ticker'] = ticker
                all_data.append(data)
            else:
                print(f"Data for {ticker} is unavailable or empty.")

        if not all_data:
            raise ValueError("No valid data fetched for any tickers.")
        
        df = pd.concat(all_data, axis=0)
        df = self.add_features(df)
        
        return df.dropna()

    def add_features(self, df):
        """Adds technical indicators and sentiment score as features."""
        #trading algs go here
        #RFC looks at algs and decides which to trade
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['Bollinger_Upper'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
        df['Bollinger_Lower'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Next_Day_Return'] = df['Close'].pct_change(1).shift(-1)
        df['Target'] = (df['Next_Day_Return'] > 0).astype(int)  # 1 if next day return is positive, else 0
        
        # Add sentiment feature 
        df['Sentiment'] = nt.HeadlineAnalyzer().get_sentiment()
        return df

    def train_model(self, df):
        """Trains a tuned RandomForestClassifier to predict buy/sell signals."""
        features = ['Close', 'Open', 'SMA_10', 'SMA_50', 'Bollinger_Upper', 'Bollinger_Lower', 'Volume_Change', 'Sentiment']
        df = df.dropna()
        X = df[features]
        y = df['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        model = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        model.fit(X_train, y_train)
        
        best_model = model.best_estimator_
        predictions = best_model.predict(X_test)
        
        #print("\nBest Model Parameters:", model.best_params_)
        #print("\nClassification Report:")
        #print(classification_report(y_test, predictions))
        
        return best_model
    
    def backtest(self, model, df):
        """Backtests the model to simulate trading performance."""
        df = df.dropna()
        features = ['Close', 'Open', 'SMA_10', 'SMA_50', 'Bollinger_Upper', 'Bollinger_Lower', 'Volume_Change', 'Sentiment']
        df['Predicted'] = model.predict(df[features])
        df['Strategy_Return'] = df['Next_Day_Return'] * df['Predicted']
        
        df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
        df['Cumulative_Buy_Hold'] = (1 + df['Next_Day_Return']).cumprod()
        
        """
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Cumulative_Strategy'], label='Strategy', linestyle='dashed')
        plt.plot(df.index, df['Cumulative_Buy_Hold'], label='Buy & Hold', linestyle='solid')
        plt.legend()
        plt.title("Backtest: Strategy vs. Buy & Sell")
        plt.show()
        """
        
        final_return = df['Cumulative_Strategy'].iloc[-1]
        buy_hold = df['Cumulative_Buy_Hold'].iloc[-1]
        #print(f"Final Strategy Return: {final_return:.2f}x")
        #print(f"Final Buy and Hold Return: {buy_hold:.2f}x")
        
        # Indicate Buy/Sell signals
        df['Signal'] = df['Predicted'].apply(lambda x: 'BUY' if x == 1 else 'SELL')
        #print("Trade Signals:")
        #print(df[['Close', 'Signal']].tail(10))  # Show last 10 predictions

        return df['Signal'].iloc[-1]


"""
if __name__ == "__main__":
    chip_companies = ['NVDA']
    rfc = RFC(chip_companies)
    
    data = rfc.prepare_data()
    model = rfc.train_model(data)
    rfc.backtest(model, data)
"""
