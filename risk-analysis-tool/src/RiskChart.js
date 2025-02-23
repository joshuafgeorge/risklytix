import React from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

const RiskChart = ({ historicalData, futureRisk }) => {
  if (!historicalData || historicalData.length === 0) {
    return <p style={{ color: "#fff", textAlign: "center" }}>No Historical Data Available</p>;
  }

  // Ensure future risk data is valid
  const validFutureRisk = futureRisk && futureRisk.length > 0 ? futureRisk : [];

  // Format historical data correctly
  const formattedData = historicalData.map((item, index) => ({
    Date: item.Date,
    Historical_Risk: item.Sharpe_Ratio ? parseFloat(item.Sharpe_Ratio) : null,
    Predicted_Risk: index < validFutureRisk.length ? parseFloat(validFutureRisk[index]) : null,
  }));

  return (
    <div style={{ width: "90%", margin: "20px auto", color: "#fff" }}>
      <h3 style={{ textAlign: "center" }}>Historical vs. Predicted Risk</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={formattedData}>
          <CartesianGrid stroke="#ccc" strokeDasharray="5 5" />
          <XAxis dataKey="Date" tick={{ fill: "#fff" }} />
          <YAxis tick={{ fill: "#fff" }} label={{ value: "Risk Score", angle: -90, position: "insideLeft", fill: "#fff" }} />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="Historical_Risk" stroke="#4caf50" name="Historical Risk" />
          <Line type="monotone" dataKey="Predicted_Risk" stroke="#f44336" name="Predicted Risk" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default RiskChart;
