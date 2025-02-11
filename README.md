# News-Based Stock Trading Model

## Overview
This project integrates financial data, news sentiment analysis, and machine learning models to predict stock trading signals. It employs Random Forest Classification, linear regression analysis, and sentiment analysis to evaluate stock trends and generate buy/sell recommendations.

## Features
- **Financial Data Analysis:** Fetches and processes stock data using Yahoo Finance (`yfinance`).
- **News Sentiment Analysis:** Retrieves and analyzes news headlines using `NewsAPI` and `VADER Sentiment Analyzer`.
- **Machine Learning Models:** Utilizes a `RandomForestClassifier` for stock movement prediction.
- **Linear Regression Modeling:** Computes trends in stock prices to indicate potential price movement.
- **Backtesting Framework:** Simulates trading strategies based on generated signals.

## Project Structure
```
├── main.py                  # Entry point; calls DataPooling to generate predictions
├── dataPooling.py           # Aggregates data from different models
├── finance_data.py          # Fetches stock price data from Yahoo Finance
├── news.py                  # Retrieves financial news headlines
├── newsTrading.py           # Performs sentiment analysis on news headlines
├── RFC.py                   # Implements the Random Forest Classification model
├── linearRegression.py      # Implements linear regression-based stock analysis
├── pyvenv.cfg               # Python virtual environment configuration file
```

## Installation
### Prerequisites
Ensure you have Python installed along with the following dependencies:
```sh
pip install yfinance newsapi-python vaderSentiment scikit-learn matplotlib pandas numpy
```

## Usage
1. **Set Up News API Key:**
   - Create a `key.txt` file and store your `NewsAPI` key inside.

2. **Run the Application:**
```sh
python main.py
```
This will fetch stock data, analyze news sentiment, and provide a buy/sell recommendation.

## How It Works
- `dataPooling.py` gathers data from:
  - `RFC.py` - Random Forest classification model
  - `newsTrading.py` - News sentiment analysis
  - `linearRegression.py` - Stock trend analysis
- Aggregated data determines whether to buy or sell a stock.
- `main.py` prints the final decision.

## Future Improvements
- Implement a GUI using `tkinter`.
- Support multiple stock tickers dynamically.
- Improve efficiency by optimizing API calls and data handling.
- Enhance backtesting for more accurate performance evaluation.

## Author
Developed by **Walter IT**.

