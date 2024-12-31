import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Define the list of stocks
stocks = ['PXLW', 'TGAN', 'NA', 'TRT', 'PRSO', 'LEDS', 'VLN', 'EMKR', 'SQNS', 'GSIT']

# Define a function to fetch and preprocess stock data
def fetch_stock_data(stock):
    try:
        data = yf.download(stock, period="1y", interval="1h")  # 1 year of hourly data
        data['Stock'] = stock
        data['Return'] = data['Close'].pct_change()
        data.dropna(inplace=True)
        return data
    except Exception as e:
        print(f"Error fetching data for {stock}: {e}")
        return None

# Fetch data for all stocks
dataframes = []
for stock in stocks:
    df = fetch_stock_data(stock)
    if df is not None and not df.empty:
        print(f"Data fetched for {stock}, head:\n{df.head()}\n")
        dataframes.append(df)
    else:
        print(f"No valid data for {stock} after fetching.")

# Combine all data into a single DataFrame
if dataframes:
    combined_data = pd.concat(dataframes, ignore_index=True)
    print(f"Combined data shape: {combined_data.shape}\nSample data:\n{combined_data.head()}")
else:
    print("No data could be fetched for any of the provided stocks. Exiting.")
    exit()

# Feature engineering
combined_data['Lag_Return'] = combined_data.groupby('Stock')['Return'].shift(1)
print(f"Data after adding Lag_Return:\n{combined_data.head()}")

combined_data['Volatility'] = combined_data.groupby('Stock')['Return'].rolling(window=5).std().reset_index(0, drop=True)
print(f"Data after adding Volatility:\n{combined_data.head()}")

combined_data.dropna(inplace=True)
print(f"Data shape after preprocessing and dropping NA: {combined_data.shape}\nSample data:\n{combined_data.head()}")

# Check if combined_data is empty
if combined_data.empty:
    print("No data available after preprocessing. Check your data sources and preprocessing steps.")
    exit()

# Prepare training data
X = combined_data[['Lag_Return', 'Volatility']]
y = (combined_data['Return'] > 0).astype(int)  # 1 if return is positive, else 0

# Check if X and y have enough samples
if len(X) == 0 or len(y) == 0:
    print("No samples available for training/testing. Ensure enough data is available after feature engineering.")
    exit()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Feature importance
importance = pd.DataFrame({'Feature': X.columns, 'Importance': clf.feature_importances_})
importance.sort_values(by='Importance', ascending=False, inplace=True)

# Plot feature importance
plt.figure(figsize=(8, 6))
plt.barh(importance['Feature'], importance['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

# Save the model for further use
import joblib
joblib.dump(clf, 'hft_model.pkl')
