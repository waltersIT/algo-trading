import yfinance as yf
import numpy as np

class FinanceData:
    def __init__(self):
        pass
        
    def fetch_financial_data(self, ticker):
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