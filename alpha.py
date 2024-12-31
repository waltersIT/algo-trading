#API KEY: 38ZWG370RS8AKVJ6

from alpha_vantage.timeseries import TimeSeries
import time

# Initialize TimeSeries with your API key
ts = TimeSeries(key='38ZWG370RS8AKVJ6', output_format='pandas')

# Function to fetch data with error handling
def fetch_data(symbol):
    try:
        data, meta_data = ts.get_intraday(symbol=symbol, interval='5min', outputsize='full')
        return data
    except Exception as e:  # Catch all exceptions
        print(f"Error fetching data: {e}")
        return None

# Fetch data for IBM
data = fetch_data('IBM')
if data is not None:
    print(data.head())
else:
    print("Failed to retrieve data.")
