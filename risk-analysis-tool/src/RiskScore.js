import { CircularProgressbar, buildStyles } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";

const RiskScore = ({ risk }) => {
  const percentage = Math.round(risk * 100); // Convert to percentage

  // Determine color based on risk level
  const getColor = (value) => {
    if (value <= 33) return "#4caf50"; // Green (Low Risk)
    if (value <= 66) return "#ffeb3b"; // Yellow (Medium Risk)
    return "#f44336"; // Red (High Risk)
  };

  return (
    <div style={{ width: "150px", margin: "0 auto" }}> {/* ✅ Make it smaller */}
      <CircularProgressbar
        value={percentage}
        text={`${percentage}%`}
        styles={buildStyles({
          textColor: "#fff",
          pathColor: getColor(percentage),
          trailColor: "#444", // Background track color
          strokeLinecap: "round",
        })}
      />
      <p style={{ textAlign: "center", color: "#fff", marginTop: "10px" }}>Risk Score</p>
    </div>
  );
};

export default RiskScore;
