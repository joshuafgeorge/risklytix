"use client";

import { useParams } from "next/navigation";

export default function StockAnalysis() {
  const { symbol } = useParams<{ symbol: string }>();

  return <h1>Analysis for {symbol}</h1>;
}
