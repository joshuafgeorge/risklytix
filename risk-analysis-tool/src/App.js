import React, { useState } from "react";
import axios from "axios";
import RiskScore from "./RiskScore";
import RiskChart from "./RiskChart";
import HistoricalGraphs from "./HistoricalGraphs";
import SentimentQuadrant from "./SentimentQuadrant"; // Updated to use the quadrant

function App() {
  const [ticker, setTicker] = useState("");
  const [riskData, setRiskData] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setRiskData(null);
    setError(null);

    try {
      const response = await axios.post("http://127.0.0.1:5000/api/calculate_risk", { ticker });

      if (response.data.risk_number) {
        setRiskData(response.data);
      } else {
        setError(response.data.error || "An error occurred.");
      }
    } catch (err) {
      setError("Failed to fetch data. Please try again.");
    }
  };

  return (
    <div style={{ textAlign: "center", padding: "20px", backgroundColor: "#212121", minHeight: "100vh", color: "#fff" }}>
      <h1>Risklytix Dashboard</h1>
      <form onSubmit={handleSubmit}>
        <input type="text" value={ticker} onChange={(e) => setTicker(e.target.value)} placeholder="Enter Ticker Symbol" />
        <button type="submit">Analyze</button>
      </form>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {riskData && (
        <>
          <RiskScore risk={riskData.risk_number} />
          <RiskChart historicalData={riskData.historical_data} futureRisk={riskData.future_risk} />
          <HistoricalGraphs historicalData={riskData.historical_data} />
          <SentimentQuadrant sentimentScore={riskData.sentiment_score} />
        </>
      )}
    </div>
  );
}

export default App;
