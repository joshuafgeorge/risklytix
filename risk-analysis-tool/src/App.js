import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [ticker, setTicker] = useState(""); // State for ticker input
  const [riskNumber, setRiskNumber] = useState(null); // State for risk number result
  const [error, setError] = useState(null); // State for error messages

  const handleSubmit = async (event) => {
    event.preventDefault();
    setRiskNumber(null); // Reset previous results
    setError(null);

    try {
      // Send ticker symbol to Flask API
      const response = await axios.post("http://127.0.0.1:5000/api/calculate_risk", { ticker });

      // If Flask responds with a risk number, display it
      if (response.data.risk_number) {
        setRiskNumber(response.data.risk_number);
      } else {
        setError(response.data.error || "An error occurred.");
      }
    } catch (err) {
      setError("Failed to fetch data. Please try again.");
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Risk Analysis Tool</h1>
        <form onSubmit={handleSubmit} className="form">
          <label htmlFor="ticker">Enter Ticker Symbol:</label>
          <input
            type="text"
            id="ticker"
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            placeholder="e.g., AAPL"
            required
          />
          <button type="submit">Analyze</button>
        </form>

        {riskNumber !== null && (
          <div className="result">
            <h2>Risk Number: {riskNumber}</h2>
          </div>
        )}

        {error && (
          <div className="error">
            <p>{error}</p>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
