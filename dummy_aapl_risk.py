import boto3
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from decimal import Decimal
import math

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
    print(f"Stock data for {ticker} added successfully!")

# Fetch historical stock data from DynamoDB
def fetch_data(table_name):
    try:
        table = dynamodb.Table(table_name)
        response = table.scan()  
        if 'Items' not in response:
            raise ValueError("No data found in DynamoDB")
        data = response['Items']
        
        df = pd.DataFrame(data)
        
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Volatility', 
                           'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'RSI_14', 'Sharpe_Ratio']
        for col in numeric_columns:
            if col in df.columns:  
                df[col] = df[col].astype(float)
        
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values(by='Date', inplace=True)

        return df
    except Exception as e:
        print(f"Error fetching data from DynamoDB: {e}")
        return None

# Check and clean NaN or Infinity values
def clean_numeric_values(value):
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None  
        else:
            return Decimal(str(value))  
    elif isinstance(value, Decimal):
        return value  
    return value  

# Calculate the risk factor using the data
def calculate_risk_factor(df):
    volatility_weight = Decimal("0.4")
    return_weight = Decimal("0.3")
    rsi_weight = Decimal("0.2")
    sharpe_ratio_weight = Decimal("0.1")

    risk_factors = []

    for i in range(len(df)):
        row = df.iloc[i]
        
        volatility = clean_numeric_values(row['Volatility']) if row['Volatility'] is not None else None
        daily_return = clean_numeric_values(abs(row['Return']))  
        rsi_diff = clean_numeric_values(row['RSI_14'] - 50)  
        sharpe_ratio = clean_numeric_values(row['Sharpe_Ratio']) if row['Sharpe_Ratio'] is not None else None

        if volatility is not None and daily_return is not None and rsi_diff is not None and sharpe_ratio is not None:
            risk_factor = (
                volatility_weight * volatility +
                return_weight * daily_return +
                rsi_weight * rsi_diff +
                sharpe_ratio_weight * (Decimal("1") - sharpe_ratio) 
            )
        else:
            risk_factor = None  

        risk_factors.append(risk_factor)

    df['Risk_Factor'] = risk_factors

    min_risk = df['Risk_Factor'].min()
    max_risk = df['Risk_Factor'].max()

    if max_risk != min_risk:
        df['Normalized_Risk_Factor'] = (df['Risk_Factor'] - min_risk) / (max_risk - min_risk)
    else:
        df['Normalized_Risk_Factor'] = Decimal("0.5")

    return df

# Update the risk factor in DynamoDB
def update_risk_factor_in_dynamodb(table_name):
    df = fetch_data(table_name)
    if df is None:
        return

    df = calculate_risk_factor(df)

    print(df[['Date', 'Risk_Factor', 'Normalized_Risk_Factor']].head())

    table = dynamodb.Table(table_name)
    
    for index, row in df.iterrows():
        item = {
            "ticker": row["ticker"],  
            "Date": row["Date"].strftime("%Y-%m-%d"),
            "Open": clean_numeric_values(row.get("Open")),
            "High": clean_numeric_values(row.get("High")),
            "Low": clean_numeric_values(row.get("Low")),
            "Close": clean_numeric_values(row.get("Close")),
            "Volume": clean_numeric_values(row.get("Volume")),
            "Return": clean_numeric_values(row.get("Return")),
            "Volatility": clean_numeric_values(row.get("Volatility")),
            "SMA_50": clean_numeric_values(row.get("SMA_50")),
            "SMA_200": clean_numeric_values(row.get("SMA_200")),
            "EMA_50": clean_numeric_values(row.get("EMA_50")),
            "EMA_200": clean_numeric_values(row.get("EMA_200")),
            "RSI_14": clean_numeric_values(row.get("RSI_14")),
            "Sharpe_Ratio": clean_numeric_values(row.get("Sharpe_Ratio")),
            "Risk_Factor": clean_numeric_values(row.get("Risk_Factor")),  
            "Normalized_Risk_Factor": clean_numeric_values(row.get("Normalized_Risk_Factor")) 
        }

        table.put_item(Item=item)

if __name__ == "__main__":
    ticker_symbol = input("Enter the ticker symbol: ").upper()
    
    table_name_for_ticker = f"{ticker_symbol}_StockData"
    
    create_table(table_name_for_ticker)
    
    fetch_and_store_data(ticker_symbol, table_name_for_ticker)
    
    update_risk_factor_in_dynamodb(table_name_for_ticker)
    
