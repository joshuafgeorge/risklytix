import boto3
import yfinance as yf
from datetime import datetime

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
            "ticker": ticker,  # Corrected key name to 'ticker'
            "Date": index.strftime("%Y-%m-%d"),
            "Open": int(row["Open"]),
            "High": int(row["High"]),
            "Low": int(row["Low"]),
            "Close": int(row["Close"]),
            "Volume": int(row["Volume"]),
        }
    )

print("Stock data added successfully!")
