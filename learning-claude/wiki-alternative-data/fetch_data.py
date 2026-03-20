"""
fetch_data.py — Run this once to populate data/raw/

Pulls:
  1. S&P 500 constituent list  →  data/raw/sp500.csv
  2. Wikipedia pageviews        →  data/raw/wiki_pageviews.csv
  3. Stock OHLCV data           →  data/raw/stock_data.csv
  4. SPY benchmark              →  data/raw/spy.csv

Date range: 2 years back from today (plenty for rolling-window analysis).
"""

import sys
import os
from datetime import datetime, timedelta

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
from sp500 import get_sp500
from wiki_pageviews import get_pageviews_bulk
from stock_data import get_stock_data_bulk, get_spy

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")

# Date range: 2 years back through yesterday (Wikimedia has a ~1 day lag)
END_DATE = datetime.today() - timedelta(days=1)
START_DATE = END_DATE - timedelta(days=730)

START_WIKI  = START_DATE.strftime("%Y%m%d")
END_WIKI    = END_DATE.strftime("%Y%m%d")
START_STOCK = START_DATE.strftime("%Y-%m-%d")
END_STOCK   = END_DATE.strftime("%Y-%m-%d")

print(f"Date range: {START_STOCK} → {END_STOCK}\n")


# ── 1. S&P 500 list ──────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Fetching S&P 500 constituent list...")
sp500 = get_sp500()
print(f"  Got {len(sp500)} companies")
sp500.to_csv(os.path.join(RAW_DIR, "sp500.csv"), index=False)
print(f"  Saved → data/raw/sp500.csv\n")


# ── 2. Wikipedia pageviews ───────────────────────────────────────────────────
print("=" * 60)
print(f"Step 2: Fetching Wikipedia pageviews for {len(sp500)} articles...")
print(f"  (This will take a few minutes — saving as we go)\n")

articles = sp500["wiki_article"].tolist()
checkpoint = os.path.join(RAW_DIR, "wiki_pageviews_checkpoint.csv")
views_df = get_pageviews_bulk(articles, START_WIKI, END_WIKI, checkpoint_path=checkpoint)

print(f"\n  Got {len(views_df):,} rows for {views_df['article'].nunique()} articles")
# Rename checkpoint to final file
import shutil
shutil.copy(checkpoint, os.path.join(RAW_DIR, "wiki_pageviews.csv"))
print(f"  Saved → data/raw/wiki_pageviews.csv\n")


# ── 3. Stock OHLCV data ──────────────────────────────────────────────────────
print("=" * 60)
print(f"Step 3: Fetching stock data for {len(sp500)} tickers...")

tickers = sp500["ticker"].tolist()
stock_df = get_stock_data_bulk(tickers, START_STOCK, END_STOCK)

print(f"\n  Got {len(stock_df):,} rows for {stock_df['ticker'].nunique()} tickers")
stock_df.to_csv(os.path.join(RAW_DIR, "stock_data.csv"), index=False)
print(f"  Saved → data/raw/stock_data.csv\n")


# ── 4. SPY benchmark ─────────────────────────────────────────────────────────
print("=" * 60)
print("Step 4: Fetching SPY benchmark...")
spy_df = get_spy(START_STOCK, END_STOCK)
print(f"  Got {len(spy_df)} rows")
spy_df.to_csv(os.path.join(RAW_DIR, "spy.csv"), index=False)
print(f"  Saved → data/raw/spy.csv\n")


# ── Summary ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("Done! Files written to data/raw/:")
for fname in ["sp500.csv", "wiki_pageviews.csv", "stock_data.csv", "spy.csv"]:
    path = os.path.join(RAW_DIR, fname)
    size_kb = os.path.getsize(path) / 1024
    print(f"  {fname:<25} {size_kb:>8.1f} KB")
