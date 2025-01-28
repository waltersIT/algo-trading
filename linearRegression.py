#plots penny stocks with linear regression
import finance_data as fd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

class LinearRegressionTrading:

    def __init__(self):
        self.finance_data = fd.FinanceData()

    def plot_stock_with_regression(self, chip_companies):
        """Fetches and plots stock data with linear regression lines."""
        plt.figure(figsize=(16, 10))

        for ticker in chip_companies:
            #print(ticker)
            data = self.finance_data.fetch_financial_data(ticker)
            if data is not None and not data.empty:
                # Prepare data for linear regression
                data = data.reset_index()
                data['Timestamp'] = data['Date'].map(pd.Timestamp.timestamp)
                X = data['Timestamp'].values.reshape(-1, 1)
                y = data['Close'].values

                # Fit linear regression model
                model = LinearRegression()
                model.fit(X, y)

                # Predict values
                y_pred = model.predict(X)

                # Plot original data
                plt.plot(data['Date'], y, label=f"{ticker} Actual")

                # Plot regression line
                plt.plot(data['Date'], y_pred, linestyle='--', label=f"{ticker} Trend")

        plt.title("Chip Companies Stock Performance with Linear Regression")
        plt.xlabel("Date")
        plt.ylabel("Stock Price (USD)")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_stock_progress(self, chip_companies):
        """Fetches and plots the progress of chip company stocks."""
        plt.figure(figsize=(12, 8))

        for ticker in chip_companies:
            data = self.finance_data.fetch_financial_data(ticker)
            if data is not None and not data.empty:
                plt.plot(data.index, data['Close'], label=ticker)

        plt.title("Chip Companies Stock Performance (Past Month)")
        plt.xlabel("Date")
        plt.ylabel("Stock Price (USD)")
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    linear = LinearRegressionTrading()
    chip_companies = ['NVDA']
    linear.plot_stock_progress(chip_companies)
    linear.plot_stock_with_regression(chip_companies)

