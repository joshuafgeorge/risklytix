import boto3
import pandas as pd
from decimal import Decimal
from datetime import datetime
import math

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb')
table_name = "AAPL_StockData"  # Replace with your DynamoDB table name
table = dynamodb.Table(table_name)

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

# Check and clean NaN or Infinity values
def clean_numeric_values(value):
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None  # Replace NaN or Infinity with None (or choose a default like 0)
        else:
            return Decimal(str(value))  # Convert valid numbers to Decimal
    elif isinstance(value, Decimal):
        return value  # Return Decimal as is
    return value  # Non-numeric values remain unchanged

# Calculate the risk factor using the data
def calculate_risk_factor(df):
    # Define weights as Decimal for compatibility
    volatility_weight = Decimal("0.4")
    return_weight = Decimal("0.3")
    rsi_weight = Decimal("0.2")
    sharpe_ratio_weight = Decimal("0.1")

    # Calculate Risk Factor
    risk_factors = []

    for i in range(len(df)):
        row = df.iloc[i]
        
        # Calculate the Risk Factor using the formula
        volatility = clean_numeric_values(row['Volatility']) if row['Volatility'] is not None else None
        daily_return = clean_numeric_values(abs(row['Return']))  # Absolute value of daily return
        rsi_diff = clean_numeric_values(row['RSI_14'] - 50)  # Difference from 50 (centered)
        sharpe_ratio = clean_numeric_values(row['Sharpe_Ratio']) if row['Sharpe_Ratio'] is not None else None

        # Risk Factor formula
        if volatility is not None and daily_return is not None and rsi_diff is not None and sharpe_ratio is not None:
            risk_factor = (
                volatility_weight * volatility +
                return_weight * daily_return +
                rsi_weight * rsi_diff +
                sharpe_ratio_weight * (Decimal("1") - sharpe_ratio)  # 1 - Sharpe ratio, lower Sharpe ratio increases risk
            )
        else:
            risk_factor = None  # If any of the factors are missing, set risk factor to None

        risk_factors.append(risk_factor)

    # Add risk factors to the dataframe
    df['Risk_Factor'] = risk_factors

    # Normalize the Risk Factor to be between 0 and 1
    min_risk = df['Risk_Factor'].min()
    max_risk = df['Risk_Factor'].max()

    # Avoid division by zero if all values are the same (check if min == max)
    if max_risk != min_risk:
        df['Normalized_Risk_Factor'] = (df['Risk_Factor'] - min_risk) / (max_risk - min_risk)
    else:
        # If min == max, assign all values a normalized value of 0.5 (midpoint)
        df['Normalized_Risk_Factor'] = Decimal("0.5")

    return df

# Update the risk factor in DynamoDB
def update_risk_factor_in_dynamodb():
    # Fetch historical stock data
    df = fetch_data()
    if df is None:
        return

    # Calculate the risk factor for each day
    df = calculate_risk_factor(df)

    # Print the DataFrame with risk factor for reference
    print(df[['Date', 'Risk_Factor', 'Normalized_Risk_Factor']].head())

    # Update the risk factor data back to DynamoDB
    for index, row in df.iterrows():
        item = {
            "ticker": row["ticker"],  # Corrected key name to 'ticker'
            "Date": row["Date"].strftime("%Y-%m-%d"),
            "Open": clean_numeric_values(row["Open"]),
            "High": clean_numeric_values(row["High"]),
            "Low": clean_numeric_values(row["Low"]),
            "Close": clean_numeric_values(row["Close"]),
            "Volume": clean_numeric_values(row["Volume"]),
            "Return": clean_numeric_values(row["Return"]),
            "Volatility": clean_numeric_values(row["Volatility"]),
            "SMA_50": clean_numeric_values(row["SMA_50"]) if row["SMA_50"] is not None else None,
            "SMA_200": clean_numeric_values(row["SMA_200"]) if row["SMA_200"] is not None else None,
            "EMA_50": clean_numeric_values(row["EMA_50"]) if row["EMA_50"] is not None else None,
            "EMA_200": clean_numeric_values(row["EMA_200"]) if row["EMA_200"] is not None else None,
            "RSI_14": clean_numeric_values(row["RSI_14"]) if row["RSI_14"] is not None else None,
            "Sharpe_Ratio": clean_numeric_values(row["Sharpe_Ratio"]) if row["Sharpe_Ratio"] is not None else None,
            "Risk_Factor": clean_numeric_values(row["Risk_Factor"]),  # Ensure it's a Decimal or None
            "Normalized_Risk_Factor": clean_numeric_values(row["Normalized_Risk_Factor"])  # Normalize and store
        }

        # Put the item in DynamoDB
        table.put_item(Item=item)

# Run the process
if __name__ == "__main__":
    update_risk_factor_in_dynamodb()
    print("Risk factor data added to DynamoDB.")
