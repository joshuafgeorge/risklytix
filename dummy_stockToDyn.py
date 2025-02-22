import boto3
import yfinance as yf
from datetime import datetime
from decimal import Decimal  # Import Decimal

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb')
table_name = "stockdat"
table = dynamodb.Table(table_name)

# Fetch stock data
ticker = "AAPL"
stock = yf.Ticker(ticker)
data = stock.history(period="1d")

# Insert data into DynamoDB
for index, row in data.iterrows():
    table.put_item(
        Item={  
            "ticker": ticker,
            "Date": index.strftime("%Y-%m-%d"),
            "Open": Decimal(str(row["Open"])),   # Convert float to Decimal
            "High": Decimal(str(row["High"])),
            "Low": Decimal(str(row["Low"])),
            "Close": Decimal(str(row["Close"])),
            "Volume": int(row["Volume"]),  # Keep Volume as int
        }
    )

print("Stock data added successfully!")
