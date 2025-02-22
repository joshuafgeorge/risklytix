import RiskScoreCard from "../components/RiskScoreCard";

const stocks = [
  { symbol: "AAPL", riskScore: 65 },
  { symbol: "TSLA", riskScore: 80 },
];

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <h1 className="text-3xl font-bold mb-6">RiskLytix Dashboard</h1>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {stocks.map((stock) => (
          <RiskScoreCard key={stock.symbol} {...stock} />
        ))}
      </div>
    </div>
  );
}
