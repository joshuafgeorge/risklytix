"use client";

import { useRouter } from "next/navigation";
import { CircularProgressbar, buildStyles } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";

interface RiskScoreCardProps {
  symbol: string;
  riskScore: number;
}

const getRiskColor = (score: number) => {
  if (score <= 30) return "#22c55e";
  if (score <= 70) return "#eab308";
  return "#ef4444";
};

const RiskScoreCard = ({ symbol, riskScore }: RiskScoreCardProps) => {
  const router = useRouter();
  const color = getRiskColor(riskScore);

  return (
    <div
      className="bg-gray-800 p-6 rounded-lg cursor-pointer hover:bg-gray-700 transition"
      onClick={() => router.push(`/analysis/${symbol}`)}
    >
      <h3 className="text-xl font-bold">{symbol}</h3>
      <CircularProgressbar
        value={riskScore}
        text={`${riskScore}`}
        styles={buildStyles({ pathColor: color, textColor: color })}
      />
    </div>
  );
};

export default RiskScoreCard;  
