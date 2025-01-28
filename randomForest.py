#Random Forest including news headlines
import finance_data
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# List of major chip company stock tickers
#chip_companies = ['PXLW', 'TRT', 'PRSO', 'LEDS', 'VLN', 'EMKR', 'SQNS', 'GSIT']
class RFC:
    def __init__(self, chip_companies):
        self.chip_companies = chip_companies
        self.finance_data = finance_data.FinanceData()

    def prepare_data(self):
        """Prepares dataset for machine learning."""
        all_data = []

        for ticker in self.chip_companies:
            #print(ticker)
            data = self.finance_data.fetch_financial_data(ticker)

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


    def train_model(self, df):
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
"""
    def plot_feature_importance(self, model, features):
        #Plots feature importance.
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(range(len(features)), importances[indices], align="center")
        plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45)
        plt.show()
"""
if __name__ == "__main__":
    chip_companies = ['NVDA']

    rfc = RFC(chip_companies)

    data = rfc.prepare_data()
    print("Dataset Sample:\n", data)
    
    model = rfc.train_model(data)
    #rfc.plot_feature_importance(model, ['P/E Ratio', 'EPS', 'Market Cap', 'Close', 'Open'])
