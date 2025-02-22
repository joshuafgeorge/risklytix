import boto3
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from decimal import Decimal

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb')
table_name = "AAPL_StockData"

# Create DynamoDB Table (if it doesn't exist)
def create_table():
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
        print("Creating table... This may take a few minutes.")
        table.wait_until_exists()
    print(f"Table '{table_name}' is ready.")

# Fetch historical stock data and compute risk metrics
def fetch_and_store_data():
    table = dynamodb.Table(table_name)
    ticker = "AAPL"
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
    risk_free_rate = 0.02 / 252  # Assume 2% annual risk-free rate converted to daily
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
    print("Stock data with no missing dates added successfully!")

# Run functions
create_table()
fetch_and_store_data()
