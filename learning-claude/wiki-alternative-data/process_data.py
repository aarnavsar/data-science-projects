"""
process_data.py — Run this after fetch_data.py to build processed outputs.

Reads from data/raw/ and writes to data/processed/:
  - spikes.csv       : pageviews with rolling stats + spike flags + ticker/sector
  - returns.csv      : daily abnormal returns + volume ratio per ticker
  - signal_table.csv : spikes joined with next-day stock outcomes (main analysis input)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
from signals import detect_spikes, compute_abnormal_returns, build_signal_table

RAW_DIR  = os.path.join(os.path.dirname(__file__), "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")


def load_raw():
    print("Loading raw data...")
    views_df = pd.read_csv(os.path.join(RAW_DIR, "wiki_pageviews.csv"), parse_dates=["date"])
    stock_df = pd.read_csv(os.path.join(RAW_DIR, "stock_data.csv"),     parse_dates=["date"])
    spy_df   = pd.read_csv(os.path.join(RAW_DIR, "spy.csv"),            parse_dates=["date"])
    sp500    = pd.read_csv(os.path.join(RAW_DIR, "sp500.csv"))
    print(f"  Pageviews : {len(views_df):,} rows, {views_df['article'].nunique()} articles")
    print(f"  Stock data: {len(stock_df):,} rows, {stock_df['ticker'].nunique()} tickers")
    print(f"  SPY       : {len(spy_df)} rows")
    return views_df, stock_df, spy_df, sp500


def main():
    views_df, stock_df, spy_df, sp500 = load_raw()

    # Article -> ticker/sector lookup
    article_meta = sp500.set_index("wiki_article")[["ticker", "company", "sector"]]

    # ── 1. Spikes ────────────────────────────────────────────────────────────
    print("\nDetecting spikes...")
    spikes_df = detect_spikes(views_df)
    spikes_df = spikes_df.join(article_meta, on="article")
    spikes_df = spikes_df.dropna(subset=["ticker"])

    n_spikes = spikes_df["is_spike"].sum()
    print(f"  {n_spikes:,} spike days across {spikes_df['article'].nunique()} articles")
    print(f"  Spike rate: {n_spikes / len(spikes_df):.1%}")

    out = os.path.join(PROC_DIR, "spikes.csv")
    spikes_df.to_csv(out, index=False)
    print(f"  Saved -> data/processed/spikes.csv  ({os.path.getsize(out)/1024:.0f} KB)")

    # ── 2. Abnormal returns ──────────────────────────────────────────────────
    print("\nComputing abnormal returns...")
    returns_df = compute_abnormal_returns(stock_df, spy_df)
    print(f"  {len(returns_df):,} rows, {returns_df['ticker'].nunique()} tickers")

    out = os.path.join(PROC_DIR, "returns.csv")
    returns_df.to_csv(out, index=False)
    print(f"  Saved -> data/processed/returns.csv  ({os.path.getsize(out)/1024:.0f} KB)")

    # ── 3. Signal table ──────────────────────────────────────────────────────
    print("\nBuilding signal table...")
    signal_df = build_signal_table(spikes_df, returns_df)
    signal_df = signal_df.merge(sp500[["ticker", "sector"]], on="ticker", how="left")

    clean = signal_df.dropna(subset=["next_day_abnormal_return", "next_day_volume_ratio"])
    print(f"  {len(signal_df):,} total rows, {len(clean):,} with full data")
    print(f"  Spike days with outcomes: {clean['is_spike'].sum():,}")

    out = os.path.join(PROC_DIR, "signal_table.csv")
    signal_df.to_csv(out, index=False)
    print(f"  Saved -> data/processed/signal_table.csv  ({os.path.getsize(out)/1024:.0f} KB)")

    print("\nDone. Files in data/processed/:")
    for fname in ["spikes.csv", "returns.csv", "signal_table.csv"]:
        path = os.path.join(PROC_DIR, fname)
        print(f"  {fname:<25} {os.path.getsize(path)/1024:>8.0f} KB")


if __name__ == "__main__":
    main()
