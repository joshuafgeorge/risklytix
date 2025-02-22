import boto3
import pandas as pd
import requests
from transformers import pipeline
from datetime import datetime
from decimal import Decimal

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb')
table_name = "AAPL_StockData"
table = dynamodb.Table(table_name)

# Load a sentiment analysis model (from Hugging Face's transformers)
sentiment_analyzer = pipeline('sentiment-analysis')

# Fetch historical stock data from DynamoDB
def fetch_data():
    try:
        response = table.scan()  # Scan retrieves all records
        if 'Items' not in response:
            raise ValueError("No data found in DynamoDB")
        data = response['Items']
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Convert numerical fields from Decimal to float
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Volatility', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'RSI_14', 'Sharpe_Ratio']
        for col in numeric_columns:
            if col in df.columns:  # Make sure column exists before conversion
                df[col] = df[col].astype(float)
        
        # Sort by date
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values(by='Date', inplace=True)

        return df
    except Exception as e:
        print(f"Error fetching data from DynamoDB: {e}")
        return None

# Fetch News Articles for Stock to derive Sentiment
def get_stock_news_for_date(ticker, date):
    news_api_url = f"https://newsapi.org/v2/everything?q={ticker}&from={date}&to={date}&apiKey= bcfe4342707c4d218a28fd14f60db5cd"
    response = requests.get(news_api_url)
    news_data = response.json()
    
    if news_data.get('status') != 'ok' or not news_data.get('articles'):
        return []  # Return empty if no articles found
    return [article['title'] for article in news_data['articles']]

# Analyze sentiment for a list of headlines
def analyze_sentiment(headlines):
    sentiment_scores = []
    for headline in headlines:
        sentiment = sentiment_analyzer(headline)
        sentiment_scores.append(sentiment[0]['label'])  # 'POSITIVE' or 'NEGATIVE'
    return sentiment_scores

# Add sentiment to each stock record based on the date
def add_sentiment_to_stock_data():
    df = fetch_data()
    if df is None:
        return

    for idx, row in df.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d')
        ticker = row['ticker']
        
        # Fetch news for the exact date of the stock record
        headlines = get_stock_news_for_date(ticker, date_str)
        
        if not headlines:  # If no headlines found, assign neutral sentiment
            sentiment_score = 0.5
        else:
            # Get sentiment scores and calculate average sentiment
            sentiment_scores = analyze_sentiment(headlines)
            sentiment_score = sentiment_scores.count('POSITIVE') / len(sentiment_scores)  # Proportion of positive sentiment
        
        # Add sentiment to the DataFrame
        df.at[idx, 'Sentiment'] = sentiment_score

    # Print the updated dataframe (or save it back to DynamoDB)
    print(df[['Date', 'Sentiment']].head())

    # Save the updated data back to DynamoDB (if needed)
    for index, row in df.iterrows():
        table.put_item(
            Item={  
                "ticker": row["ticker"],  # Corrected key name to 'ticker'
                "Date": row["Date"].strftime("%Y-%m-%d"),
                "Open": row["Open"],
                "High": row["High"],
                "Low": row["Low"],
                "Close": row["Close"],
                "Volume": row["Volume"],
                "Return": row["Return"],
                "Volatility": row["Volatility"],
                "SMA_50": row["SMA_50"],
                "SMA_200": row["SMA_200"],
                "EMA_50": row["EMA_50"],
                "EMA_200": row["EMA_200"],
                "RSI_14": row["RSI_14"],
                "Sharpe_Ratio": row["Sharpe_Ratio"],
                "Sentiment": row["Sentiment"],  # Sentiment data
            }
        )

# Run the process
if __name__ == "__main__":
    add_sentiment_to_stock_data()
    print("Sentiment data added to stock records.")
