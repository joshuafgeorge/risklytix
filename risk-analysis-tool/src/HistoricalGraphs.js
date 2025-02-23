import React, { useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

const HistoricalGraphs = ({ historicalData }) => {
  const [timeRange, setTimeRange] = useState("1Y"); // Default to 1 year

  // Filter historical data based on selected time range
  const filterDataByTimeRange = (data, range) => {
    const now = new Date();
    const cutoff = new Date();
    switch (range) {
      case "1W":
        cutoff.setDate(now.getDate() - 7);
        break;
      case "1M":
        cutoff.setMonth(now.getMonth() - 1);
        break;
      case "3M":
        cutoff.setMonth(now.getMonth() - 3);
        break;
      case "1Y":
        cutoff.setFullYear(now.getFullYear() - 1);
        break;
      default:
        return data;
    }
    return data.filter((item) => new Date(item.Date) >= cutoff);
  };

  const filteredData = filterDataByTimeRange(historicalData, timeRange);

  return (
    <div style={{ width: "90%", margin: "20px auto", color: "#fff" }}>
      <h3 style={{ textAlign: "center" }}>Stock Indicator - Sharpe Ratio</h3>

      {/* Time Range Selection */}
      <div style={{ textAlign: "center", marginBottom: "10px" }}>
        {["1W", "1M", "3M", "1Y"].map((range) => (
          <button
            key={range}
            onClick={() => setTimeRange(range)}
            style={{
              margin: "5px",
              padding: "8px",
              background: timeRange === range ? "#ffeb3b" : "#444",
              color: "#000",
              border: "none",
              cursor: "pointer",
            }}
          >
            {range}
          </button>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={filteredData}>
          <CartesianGrid stroke="#ccc" strokeDasharray="5 5" />
          <XAxis dataKey="Date" tick={{ fill: "#fff" }} />
          <YAxis tick={{ fill: "#fff" }} label={{ value: "Sharpe Ratio", angle: -90, position: "insideLeft", fill: "#fff" }} />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="Sharpe_Ratio" stroke="#4caf50" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default HistoricalGraphs;
