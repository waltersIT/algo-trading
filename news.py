from newsapi import NewsApiClient
import json
class News:
    def __init__(self):
        pass

    def get_news():

        with open('key.txt', 'r') as file:
            API_KEY = file.read().strip()

        headlines = []
        # Initialize
        newsapi = NewsApiClient(api_key=API_KEY)

        # Get top headlines from a specific source
        top_headlines = newsapi.get_top_headlines(sources="the-wall-street-journal")
        
        # Print article titles
        for article in top_headlines["articles"][:5]:
            headlines.append(article['title'])
        return headlines
