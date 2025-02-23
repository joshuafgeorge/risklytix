from flask import Flask, request, jsonify
from flask_cors import CORS
import risklytic_final  # Import risk analysis functions
import numpy as np
import pandas as pd 

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
#lstm_model = risklytic_final.lstm_model

# Route for calculating the risk number
@app.route('/api/calculate_risk', methods=['POST'])
def calculate_risk():
    print("flask received request")
    ticker_symbol = request.json.get('ticker')
    print(f"Received ticker: {ticker_symbol}")  # Log the ticker symbol

    if not ticker_symbol:   
        return jsonify({'error': 'Ticker symbol is required'}), 400

    try:
        #

        # Run your analysis script
        print("running main script")
        risk_number, historical_data, sentiment_score, future_risk = risklytic_final.main(ticker_symbol)
        #print(f"Calculated risk number: {risk_number}")  # Log the risk number


        def clean_value(value):
            if isinstance(value, np.generic):
                return float(value)  # Convert np.float32/64 → Python float
            if pd.isna(value):  
                return None  # Convert NaN → null (for JSON)
            return value
        
        historical_data_cleaned = [
            {"Date": item["Date"], "Sharpe_Ratio": clean_value(item["Sharpe_Ratio"])}
            for item in historical_data
        ]

        # Return the risk number as a JSON response
        return jsonify({
            'risk_number': float(risk_number) if isinstance(risk_number, np.generic) else risk_number,
            'historical_data': historical_data_cleaned,
            'sentiment_score': float(sentiment_score) if isinstance(sentiment_score, np.generic) else sentiment_score,
            'future_risk': [float(r) if isinstance(r, np.generic) else r for r in future_risk]
        })

    except Exception as e:
        print(f"Error: {e}")  # Log any errors in Flask
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🔄 Starting Flask server...")
    lstm_model = risklytic_final.load_lstm_model()
    app.run(debug=True)
