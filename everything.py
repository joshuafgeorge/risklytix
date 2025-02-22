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
import joblib

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb')

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
def fetch_and_store_data(ticker, table_name):
    table = dynamodb.Table(table_name)
    stock = yf.Ticker(ticker)
    data = stock.history(period="5y")  # Fetch last 5 years of data

    # Ensure no missing dates
    full_date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
    data = data.reindex(full_date_range)  # Reindex to include all dates
    data.fillna(method="ffill", inplace=True)  # Forward-fill missing data

    # Compute financial indicators
    data["Return"] = data["Close"].pct_change() * 100  # Daily % return
    data["Volatility"] = data["Return"].rolling(window=21).std()  # 21-day rolling volatility
    data["SMA_50"] = data["Close"].rolling(window=50).mean()  # 50-day SMA
    data["SMA_200"] = data["Close"].rolling(window=200).mean()  # 200-day SMA
    data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()  # 50-day EMA
    data["EMA_200"] = data["Close"].ewm(span=200, adjust=False).mean()  # 200-day EMA

    # RSI Calculation
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    data["Upper_BB"] = data["SMA_50"] + (data["Close"].rolling(20).std() * 2)
    data["Lower_BB"] = data["SMA_50"] - (data["Close"].rolling(20).std() * 2)

    # Sharpe Ratio (Risk-adjusted return)
    risk_free_rate = 0.02 / 252  
    data["Sharpe_Ratio"] = (data["Return"] - risk_free_rate) / data["Volatility"]

    # Drop NaN values from indicators
    data.dropna(inplace=True)

    # Store data in DynamoDB
    with table.batch_writer() as batch:
        for index, row in data.iterrows():
            batch.put_item(
                Item={  
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
            )
    print(f"Stock data for {ticker} added successfully!")

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
    
    df = df[features].fillna(0)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)
        joblib.dump(scaler, f'scaler.pkl')
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
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(50))
    model.add(Dense(1))  

    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit(X, X[:, -1, -1], epochs=epochs, batch_size=batch_size)

    model.save('lstm_risk_model.h5')

# Test the trained LSTM model and predict risk factor for the current day
def predict_current_day(table_name):
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

# Main function to execute everything end-to-end for a given ticker symbol
def main():
    ticker_symbol = input("Enter the ticker symbol: ").upper()
    
    table_name_for_ticker = f"{ticker_symbol}_StockData"
    
    create_table(table_name_for_ticker)
    
    fetch_and_store_data(ticker_symbol, table_name_for_ticker)
    
    print("Fetching preprocessed training dataset...")
    
    df_train = fetch_data(table_name_for_ticker)
    
    if df_train is None or len(df_train) < 10:
        print("Not enough training data available.")
        return
    
    X_train = preprocess_data(df_train)

    print("Training LSTM model...")
    
    train_lstm(X_train)

    print("\nTesting the trained model...")
    
    predict_current_day(table_name_for_ticker)

if __name__ == "__main__":
    main()
