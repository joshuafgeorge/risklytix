from flask import Flask, request, jsonify
from flask_cors import CORS
import risklytic_final  # Import your risk analysis script

app = Flask(__name__)
CORS(app)  # Enable cross-origin resource sharing

# Route for calculating the risk number
@app.route('/api/calculate_risk', methods=['POST'])
def calculate_risk():
    ticker_symbol = request.json.get('ticker')
    print(f"Received ticker: {ticker_symbol}")  # Log the ticker symbol

    if not ticker_symbol:   
        return jsonify({'error': 'Ticker symbol is required'}), 400

    try:
        # Run your analysis script
        risk_number = risklytic_final.main(ticker_symbol)
        print(f"Calculated risk number: {risk_number}")  # Log the risk number

        # Ensure risk_number is a valid float
        risk_number = float(risk_number)

        # Return the risk number as a JSON response
        return jsonify({'risk_number': risk_number})

    except Exception as e:
        print(f"Error: {e}")  # Log any errors in Flask
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
