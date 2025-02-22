import boto3
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
import os

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
            'EMA_50', 'EMA_200', 'RSI_14', 'Sharpe_Ratio', 
            'Risk_Factor'
        ]
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

# Preprocess and normalize the data
def preprocess_data(df, sequence_length=10):
    # Select relevant features for training
    features = [
        "Open", "High", "Low", "Close", "Volume", 
        "Return", "Volatility", "SMA_50", "SMA_200", 
        "EMA_50", "EMA_200", "RSI_14", "Sharpe_Ratio"
    ]
    
    # Handle missing values (fill with 0 or another strategy)
    df = df[features].fillna(0)

    # Normalize the features using Min-Max Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Save the scaler for later use during inference
    joblib.dump(scaler, 'scaler.pkl')

    # Create sequences of data for LSTM input
    def create_sequences(data, sequence_length):
        sequences = []
        for i in range(len(data) - sequence_length):
            seq = data[i:i + sequence_length]
            sequences.append(seq)
        return np.array(sequences)

    X = create_sequences(scaled_data, sequence_length)
    
    return X, scaler

# Build and train the LSTM model
def train_lstm(X, epochs=20, batch_size=32):
    # Define the LSTM model architecture
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(50))
    model.add(Dense(1))  # Output layer (predicting Risk_Factor)

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model (using all data since this is unsupervised for now)
    history = model.fit(X, X[:, -1, -1], epochs=epochs, batch_size=batch_size)

    # Save the trained model to a file
    model.save('lstm_risk_model.h5')

    return model

# Main function to fetch data, preprocess it, and train the LSTM model
def main():
    print("Fetching data from DynamoDB...")
    df = fetch_data()
    
    if df is None or df.empty:
        print("No data available to train.")
        return

    print("Preprocessing and normalizing data...")
    sequence_length = 10  # Use last 10 days as input for each prediction
    X, scaler = preprocess_data(df, sequence_length=sequence_length)

    print("Training LSTM model...")
    train_lstm(X)

    print("Training complete. Model and scaler saved.")

if __name__ == "__main__":
    main()
