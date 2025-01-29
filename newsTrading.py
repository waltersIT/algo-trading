from news import News
import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class HeadlineAnalyzer:
    # Define a stock-related headline
    def __init__(self):
        pass

    def get_headlines(self):
        # Initialize the sentiment analyzer
        new_headlines = News.get_news()
        existing_headlines = []
        headlines = []
        with open('headlines.txt', 'r') as file:
            existing_headlines = file.read().splitlines()

        #print(f"new heads: {new_headlines}")
        #print(f"Existing: {existing_headlines}")

        with open('headlines.txt', 'a') as file:
            for i in new_headlines:
                if i not in existing_headlines:
                    file.write(i + "\n")

        with open('headlines.txt', 'r') as file:
            headlines.append(file.read())
        return headlines
    def set_sentiment(self, headlines):
        analyzer = SentimentIntensityAnalyzer()
        # Get sentiment scores
        for i in headlines:
            sentiment_scores = analyzer.polarity_scores(i)
        #print(sentiment_scores['compound'])
        # Assign weight based on sentiment
        if sentiment_scores['compound'] > 0.05:
            decision = "Buy"
            weight = sentiment_scores['compound'] * 10  # Scale weight based on sentiment score
        elif sentiment_scores['compound'] < -0.05:
            decision = "Sell"
            weight = sentiment_scores['compound'] * 10
        else:
            decision = "Hold"
            weight = 0

        # Output the decision and weight
        #print(f"Decision: {decision}")
        #print(f"Weight: {weight}")
        
        return weight
        #use weight as the feature in RFC
    def get_sentiment(self):
        headlines = self.get_headlines()
        weight = self.set_sentiment(headlines)
        return weight
    