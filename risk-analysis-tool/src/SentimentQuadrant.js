import React from "react";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from "recharts";

const SentimentQuadrant = ({ sentimentScore }) => {
  if (sentimentScore === undefined || sentimentScore === null) {
    return null; // Don't render anything if no data
  }

  // Normalize sentiment score (-1 to 1)
  const normalizedScore = Math.max(-1, Math.min(1, sentimentScore));

  // Determine X, Y position for quadrant mapping
  const xPos = normalizedScore; // X-axis: Sentiment strength
  const yPos = Math.abs(normalizedScore); // Y-axis: Sentiment confidence

  // Determine sentiment category
  let sentimentLabel = "Neutral";
  if (normalizedScore > 0.2) sentimentLabel = "Bullish Cautious";
  if (normalizedScore > 0.5) sentimentLabel = "Bullish Extreme";
  if (normalizedScore < -0.2) sentimentLabel = "Bearish Cautious";
  if (normalizedScore < -0.5) sentimentLabel = "Bearish Extreme";

  const data = [{ x: xPos, y: yPos, label: "Current Sentiment" }];

  return (
    <div style={{ width: "90%", margin: "20px auto", textAlign: "center", color: "#fff" }}>
      <h3>Sentiment Overview</h3>
      <ResponsiveContainer width="100%" height={250}>
        <ScatterChart>
          <CartesianGrid stroke="#ccc" />
          <XAxis type="number" dataKey="x" domain={[-1, 1]} tick={{ fill: "#fff" }} label={{ value: "Bearish (-) → Bullish (+)", position: "insideBottom", fill: "#fff" }} />
          <YAxis type="number" dataKey="y" domain={[0, 1]} tick={{ fill: "#fff" }} label={{ value: "Sentiment Strength", position: "insideLeft", angle: -90, fill: "#fff" }} />
          <Scatter data={data} fill={(sentimentLabel.includes("Bullish")) ? "#4caf50" : "#f44336"} />
        </ScatterChart>
      </ResponsiveContainer>
      <h4 style={{ color: (sentimentLabel.includes("Bullish")) ? "#4caf50" : "#f44336" }}>
        Sentiment: {sentimentLabel}
      </h4>
    </div>
  );
};

export default SentimentQuadrant;
