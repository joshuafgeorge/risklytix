import boto3
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from decimal import Decimal
import math
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
import os
import joblib
import requests

MODEL_PATH = "lstm_risk_model.h5"

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb')

API_KEY = "SZJJ5RXDS60BLZP7"

def get_stock_news(symbol):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    if "feed" in data:
        news_articles = []
        for article in data["feed"]:
            news_articles.append({
                "title": article["title"],
                "url": article["url"],
                "sentiment": article.get("overall_sentiment_score", "N/A"),
                "source": article.get("source", "Unknown")
            })
        return news_articles
    else:
        return {"error": "No news data found or API limit reached."}

# Create DynamoDB Table (if it doesn't exist)
def create_table(table_name):
    existing_tables = [table.name for table in dynamodb.tables.all()]
    if table_name not in existing_tables:
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {'AttributeName': 'Date', 'KeyType': 'HASH'},  # Partition Key
                {'AttributeName': 'ticker', 'KeyType': 'RANGE'}  # Sort Key
            ],
            AttributeDefinitions=[
                {'AttributeName': 'Date', 'AttributeType': 'S'},
                {'AttributeName': 'ticker', 'AttributeType': 'S'}
            ],
            ProvisionedThroughput={'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5}
        )

        print(f"Creating table '{table_name}'... This may take a few minutes.")
        table.wait_until_exists()
        print(f"Table '{table_name}' is ready.")

# Fetch historical stock data and compute risk metrics
import time  # ✅ Import time to measure execution time

def fetch_and_store_data(ticker, table_name):
    print(f"✅ Entering fetch_and_store_data() for {ticker}")

    table = dynamodb.Table(table_name)

    print(f"🔄 Fetching stock data from Yahoo Finance for {ticker}...")
    stock = yf.Ticker(ticker)

    try:
        start_time = time.time()  # Track time
        data = stock.history(period="5y")  # Fetch last 5 years of data
        print(f"✅ Stock data retrieved: {len(data)} rows (took {time.time() - start_time:.2f} sec)")
    except Exception as e:
        print(f"❌ Error fetching stock data: {e}")
        return

    if data.empty:
        print(f"❌ No stock data found for {ticker}")
        return

    print("🔄 Ensuring no missing dates...")
    try:
        full_date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
        data = data.reindex(full_date_range)
        data.ffill(inplace=True)  # ✅ This sometimes causes issues
        print("✅ Missing dates handled successfully")
    except Exception as e:
        print(f"❌ Error handling missing dates: {e}")
        return

    print("🔄 Computing financial indicators...")
    try:
        data["Return"] = data["Close"].pct_change() * 100
        data["Volatility"] = data["Return"].rolling(window=21).std()
        print(data["Volatility"])
        data["SMA_50"] = data["Close"].rolling(window=50).mean()
        print(data["SMA_50"])
        data["SMA_200"] = data["Close"].rolling(window=200).mean()
        data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()
        data["EMA_200"] = data["Close"].ewm(span=200, adjust=False).mean()
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data["RSI_14"] = 100 - (100 / (1 + rs))
        print(data["RSI_14"])
        data["Upper_BB"] = data["SMA_50"] + (data["Close"].rolling(20).std() * 2)
        data["Lower_BB"] = data["SMA_50"] - (data["Close"].rolling(20).std() * 2)
        risk_free_rate = 0.02 / 252
        data["Sharpe_Ratio"] = (data["Return"] - risk_free_rate) / data["Volatility"]
        print("✅ Financial indicators computed")
    except Exception as e:
        print(f"❌ Error computing financial indicators: {e}")
        return

    print("🔄 Dropping NaN values...")
    try:
        data.dropna(inplace=True)
        print("✅ NaN values dropped")
    except Exception as e:
        print(f"❌ Error dropping NaN values: {e}")
        return

    print("🔄 Storing data in DynamoDB...")

    try:
        BATCH_SIZE = 500  # ✅ Process in smaller chunks
        items = []

        for index, row in data.iterrows():
            item = {
                "ticker": ticker,
                "Date": index.strftime("%Y-%m-%d"),
                "Open": Decimal(str(row["Open"])),
                "High": Decimal(str(row["High"])),
                "Low": Decimal(str(row["Low"])),
                "Close": Decimal(str(row["Close"])),
                "Volume": int(row["Volume"]),
                "Return": Decimal(str(row["Return"])),
                "Volatility": Decimal(str(row["Volatility"])),
                "SMA_50": Decimal(str(row["SMA_50"])),
                "SMA_200": Decimal(str(row["SMA_200"])),
                "EMA_50": Decimal(str(row["EMA_50"])),
                "EMA_200": Decimal(str(row["EMA_200"])),
                "RSI_14": Decimal(str(row["RSI_14"])),
                "Upper_BB": Decimal(str(row["Upper_BB"])),
                "Lower_BB": Decimal(str(row["Lower_BB"])),
                "Sharpe_Ratio": Decimal(str(row["Sharpe_Ratio"]))
            }
            
            items.append(item)

            # ✅ Write in batches of 500 records
            if len(items) >= BATCH_SIZE:
                with table.batch_writer() as batch:
                    for i in items:
                        batch.put_item(Item=i)
                print(f"✅ Stored {len(items)} records in DynamoDB...")
                items = []  # ✅ Reset the list after storing

        # ✅ Store any remaining records
        if items:
            with table.batch_writer() as batch:
                for i in items:
                    batch.put_item(Item=i)
            print(f"✅ Stored final {len(items)} records in DynamoDB...")

        print(f"✅ Stock data for {ticker} stored successfully in DynamoDB!")

    except Exception as e:
        print(f"❌ Error storing data in DynamoDB: {e}")

# Fetch historical stock data from DynamoDB for training or prediction
def fetch_data(table_name):
    try:
        table = dynamodb.Table(table_name)
        response = table.scan()
        if 'Items' not in response:
            raise ValueError("No data found in DynamoDB")
        df = pd.DataFrame(response['Items'])
        numeric_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Return', 'Volatility', 'SMA_50', 'SMA_200',
            'EMA_50', 'EMA_200', 'RSI_14', 'Sharpe_Ratio'
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(float)

        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values(by='Date', inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data from DynamoDB: {e}")
        return None

# Preprocess and normalize the dataset for training or prediction
def preprocess_data(df, scaler=None, sequence_length=10):
    features = [
        "Open", "High", "Low", "Close", "Volume",
        "Return", "Volatility", "SMA_50", "SMA_200",
        "EMA_50", "EMA_200", "RSI_14", "Sharpe_Ratio"
    ]

    df = df[features].ffill()
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)
        joblib.dump(scaler, 'scaler.pkl')
    else:
        scaled_data = scaler.transform(df)

    def create_sequences(data, sequence_length):
        sequences = []
        for i in range(len(data) - sequence_length):
            seq = data[i:i + sequence_length]
            sequences.append(seq)
        return np.array(sequences)

    X = create_sequences(scaled_data, sequence_length)
    return X

# Train the LSTM model on the preprocessed dataset
def train_lstm(X, epochs=20, batch_size=32):
    if not os.path.exists(MODEL_PATH):
        print("training a new LSTM model...")
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        history = model.fit(X, X[:, -1, -1], epochs=epochs, batch_size=batch_size)
        model.save('lstm_risk_model.h5')
        print(f"Model trained and saved")
    else:
        print(f"Model already exists, skipping training")

def load_lstm_model():
    if os.path.exists(MODEL_PATH):
        print(f"Loading pre-trained model from {MODEL_PATH}")
        return load_model(MODEL_PATH)
    else:
        print("No trained model found! You may need to train it first.")
        return None

def predict_future_risk(table_name, ticker_symbol, days=7):
    df = fetch_data(table_name)
    if df is None or len(df) < 10:
        return None

    scaler = joblib.load('scaler.pkl')
    X_test = preprocess_data(df, scaler=scaler)
    model = load_model('lstm_risk_model.h5')

    future_risk = []
    last_input = X_test[-1]  # Get last known data point
    for _ in range(days):
        prediction = model.predict(np.expand_dims(last_input, axis=0)).flatten()[0]
        future_risk.append(prediction)

        # Shift input sequence to include new prediction
        last_input = np.roll(last_input, shift=-1, axis=0)
        last_input[-1, -1] = prediction  # Update with predicted value

    return future_risk

# Test the trained LSTM model and predict risk factor for the current day
def predict_current_day_with_sentiment(table_name, ticker_symbol):
    print("Fetching and preprocessing test dataset...")
    df_test = fetch_data(table_name)
    if df_test is None or len(df_test) < 10:
        print("Not enough test data available for prediction.")
        return

    scaler = joblib.load('scaler.pkl')
    X_test = preprocess_data(df_test, scaler=scaler)

    print("Loading trained LSTM model...")
    model = load_model('lstm_risk_model.h5')

    prediction_normalized = model.predict(X_test[-1:]).flatten()[0]
    print("\nPredicted Risk Factor (Normalized) for Current Day:")
    print(prediction_normalized)

    # Fetch sentiment data
    print("\nFetching sentiment data for ticker:", ticker_symbol)
    news_data = get_stock_news(ticker_symbol)

    if "error" in news_data:
        print(news_data["error"])
        sentiment_multiplier = 1  # Default to neutral multiplier if no sentiment data
    else:  # Calculate average sentiment score
        sentiment_scores = [
            float(article["sentiment"]) for article in news_data if article["sentiment"] != "N/A"
        ]

        if sentiment_scores:
            avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores)
            print(f"Average Sentiment Score: {avg_sentiment_score}")
            # Define a multiplier range based on sentiment (e.g., 0.8 to 1.2)
            sentiment_multiplier = 1 + avg_sentiment_score / 10  # Adjust scaling as needed
        else:
            print("No valid sentiment scores found.")
            sentiment_multiplier = 1  # Default to neutral multiplier

    # Adjust prediction using sentiment multiplier
    adjusted_prediction = prediction_normalized * sentiment_multiplier
    print("\nAdjusted Risk Factor with Sentiment Multiplier:")
    print(adjusted_prediction)
    return adjusted_prediction

# Main function to execute everything end-to-end for a given ticker symbol
def main(ticker_symbol):
    print(f"✅ Starting main() for {ticker_symbol}")

    table_name_for_ticker = f"{ticker_symbol}_StockData"
    print(f"🔄 Creating table: {table_name_for_ticker}")
    create_table(table_name_for_ticker)

    print(f"🔄 Fetching and storing stock data for {ticker_symbol}...")
    fetch_and_store_data(ticker_symbol, table_name_for_ticker)
    print(f"✅ Stock data fetched and stored for {ticker_symbol}")

    print("🔄 Fetching preprocessed training dataset...")
    df_train = fetch_data(table_name_for_ticker)
    if df_train is None or len(df_train) < 10:
        print("❌ Not enough training data available.")
        return None, None, None, None
    print(f"✅ Training data fetched: {len(df_train)} rows")

    print("🔄 Preprocessing training dataset...")
    X_train = preprocess_data(df_train)
    print(f"✅ Preprocessed training dataset: {X_train.shape}")

    print("🔄 Training LSTM model...")
    train_lstm(X_train)
    print("✅ LSTM model training complete")

    print("\n🔄 Testing the trained model with sentiment adjustment...")
    risk_number = predict_current_day_with_sentiment(table_name_for_ticker, ticker_symbol)
    print(f"✅ Predicted risk number: {risk_number}")

    future_risk = predict_future_risk(table_name_for_ticker, ticker_symbol)
    print(f"✅ Predicted future risk: {future_risk}")

    print("🔄 Converting historical data to JSON format...")
    df_train['Date'] = df_train['Date'].astype(str)
    historical_data = df_train[['Date', 'Sharpe_Ratio']].to_dict(orient='records')
    print("✅ Historical data conversion complete")

    print("🔄 Fetching sentiment data...")
    sentiment_data = get_stock_news(ticker_symbol)
    if "error" in sentiment_data:
        sentiment_score = 0
        print("❌ Sentiment fetch failed, setting score to 0")
    else:
        sentiment_scores = [float(article["sentiment"]) for article in sentiment_data if article["sentiment"] != "N/A"]
        sentiment_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        print(f"✅ Calculated sentiment score: {sentiment_score}")

    print("✅ Returning final results from main()")
    return risk_number, historical_data, sentiment_score, future_risk

#lstm_model =  load_lstm_model()
if __name__ == "__main__":
    main()
