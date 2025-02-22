import boto3
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib

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
        numeric_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume', 
            'Return', 'Volatility', 'SMA_50', 'SMA_200', 
            'EMA_50', 'EMA_200', 'RSI_14', 'Sharpe_Ratio'
        ]
        for col in numeric_columns:
            if col in df.columns:  # Make sure column exists before conversion
                df[col] = df[col].astype(float)
        
        # Sort by date (most recent last)
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values(by='Date', inplace=True)

        return df
    except Exception as e:
        print(f"Error fetching data from DynamoDB: {e}")
        return None

# Preprocess and normalize the data for prediction
def preprocess_data(df, scaler, sequence_length=10):
    # Select relevant features for prediction
    features = [
        "Open", "High", "Low", "Close", "Volume", 
        "Return", "Volatility", "SMA_50", "SMA_200", 
        "EMA_50", "EMA_200", "RSI_14", "Sharpe_Ratio"
    ]
    
    # Handle missing values (fill with 0 or another strategy)
    df = df[features].fillna(0)

    # Normalize the features using the loaded scaler
    scaled_data = scaler.transform(df)

    # Create sequences of data for LSTM input
    def create_sequences(data, sequence_length):
        sequences = []
        for i in range(len(data) - sequence_length):
            seq = data[i:i + sequence_length]
            sequences.append(seq)
        return np.array(sequences)

    X = create_sequences(scaled_data, sequence_length)
    
    return X

# Load model and scaler, and make a single prediction for the current day
def predict_current_day():
    print("Fetching data from DynamoDB...")
    df = fetch_data()
    
    if df is None or df.empty:
        print("No data available for prediction.")
        return

    print("Loading scaler and model...")
    scaler = joblib.load('scaler.pkl')  # Load the saved MinMaxScaler
    model = load_model('lstm_risk_model.h5')  # Load the trained LSTM model

    print("Preprocessing data...")
    sequence_length = 10  # Use last 10 days as input for each prediction
    
    # Ensure we have at least `sequence_length` rows of data to create a valid sequence
    if len(df) < sequence_length:
        print(f"Not enough data to make a prediction. At least {sequence_length} rows are required.")
        return
    
    X = preprocess_data(df, scaler, sequence_length=sequence_length)

    print("Making prediction for the current day...")
    current_day_prediction = model.predict(X[-1:])  # Use only the last sequence

    # Output only the predicted risk factor (normalized) for the current day
    predicted_risk_factor_normalized = current_day_prediction.flatten()[0]
    
    print("\nPredicted Risk Factor (Normalized) for Current Day:")
    print(predicted_risk_factor_normalized)

# Main function to run the prediction process
if __name__ == "__main__":
    predict_current_day()
