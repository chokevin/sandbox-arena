#!/usr/bin/env python3
"""Download and prepare real financial data for sandbox-arena.

Downloads historical OHLCV data for commodities, equities, and ETFs
via Yahoo Finance. Outputs clean CSVs that both sandbox-arena and
alpha-transformer can use.

Usage:
    python data/fetch_data.py                    # download all
    python data/fetch_data.py --symbols GC=F CL=F  # specific symbols
    python data/fetch_data.py --years 5          # 5 years of history
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# Default universe: commodities + major equities + sector ETFs
UNIVERSE = {
    # Commodities (futures)
    "gold": "GC=F",
    "oil": "CL=F",
    "silver": "SI=F",
    "natgas": "NG=F",
    "wheat": "ZW=F",
    "corn": "ZC=F",
    "copper": "HG=F",

    # Major equities
    "aapl": "AAPL",
    "msft": "MSFT",
    "googl": "GOOGL",
    "amzn": "AMZN",
    "nvda": "NVDA",
    "tsla": "TSLA",
    "meta": "META",

    # Sector ETFs
    "spy": "SPY",
    "qqq": "QQQ",
    "gld": "GLD",
    "uso": "USO",
    "tlt": "TLT",
    "vxx": "^VIX",
}


def fetch_symbol(symbol: str, years: int = 3) -> pd.DataFrame:
    """Download OHLCV data for a single symbol."""
    end = datetime.now()
    start = end - timedelta(days=years * 365)

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"))

    if df.empty:
        return df

    # Clean columns
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    df.index = df.index.strftime("%Y-%m-%d")

    # Add derived features
    df["return_1d"] = df["close"].pct_change()
    df["return_5d"] = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(20)
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_50"] = df["close"].rolling(50).mean()
    df["volatility_20d"] = df["return_1d"].rolling(20).std() * (252 ** 0.5)

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    # Bollinger Bands
    df["bb_upper"] = df["ma_20"] + 2 * df["close"].rolling(20).std()
    df["bb_lower"] = df["ma_20"] - 2 * df["close"].rolling(20).std()
    df["bb_position"] = (df["close"] - df["ma_20"]) / (
        df["close"].rolling(20).std() * 2 + 1e-10)

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()

    # Drop NaN rows from rolling calcs
    df = df.dropna()

    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch financial data")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Specific Yahoo Finance symbols to download")
    parser.add_argument("--years", type=int, default=3,
                        help="Years of history to download")
    parser.add_argument("--output-dir", default="data",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.symbols:
        universe = {s: s for s in args.symbols}
    else:
        universe = UNIVERSE

    print(f"Fetching {len(universe)} symbols, {args.years} years of data...")
    print(f"Output: {output_dir}/\n")

    metadata = {}

    for name, symbol in universe.items():
        print(f"  {name:10s} ({symbol})...", end=" ", flush=True)
        try:
            df = fetch_symbol(symbol, years=args.years)
            if df.empty:
                print("NO DATA")
                continue

            csv_path = output_dir / f"{name}.csv"
            df.to_csv(csv_path)

            metadata[name] = {
                "symbol": symbol,
                "rows": len(df),
                "start": df.index[0],
                "end": df.index[-1],
                "columns": list(df.columns),
                "close_min": round(df["close"].min(), 2),
                "close_max": round(df["close"].max(), 2),
            }
            print(f"{len(df)} rows ({df.index[0]} → {df.index[-1]})")

        except Exception as e:
            print(f"ERROR: {e}")

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Saved {len(metadata)} datasets to {output_dir}/")
    print(f"  Metadata: {meta_path}")

    # Summary
    total_rows = sum(m["rows"] for m in metadata.values())
    print(f"  Total rows: {total_rows:,}")


if __name__ == "__main__":
    main()
