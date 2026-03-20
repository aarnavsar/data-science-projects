import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from wiki_pageviews import get_pageviews_bulk
from stock_data import get_stock_data_bulk, get_spy


def detect_spikes(views_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag days where Wikipedia pageviews spike abnormally high.

    Uses a 30-day rolling window per article. A spike is defined as:
        views > rolling_mean + 2 * rolling_std

    Args:
        views_df: DataFrame with columns: date, article, views

    Returns:
        Same rows + columns: rolling_mean, rolling_std, is_spike
    """
    df = views_df.copy().sort_values(["article", "date"])

    df["rolling_mean"] = (
        df.groupby("article")["views"]
        .transform(lambda s: s.rolling(30, min_periods=10).mean())
    )
    df["rolling_std"] = (
        df.groupby("article")["views"]
        .transform(lambda s: s.rolling(30, min_periods=10).std())
    )

    df["z_score"] = (df["views"] - df["rolling_mean"]) / df["rolling_std"]
    df["is_spike"] = df["z_score"] > 2

    # Spike tier — only meaningful on is_spike days (NaN for non-spike rows)
    df["spike_tier"] = pd.cut(
        df["z_score"],
        bins=[2, 3, 5, float("inf")],
        labels=["weak", "medium", "strong"],
        right=False,
    )

    return df.reset_index(drop=True)


def compute_abnormal_returns(stock_df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily abnormal returns (stock return minus SPY return).

    Also computes a 30-day rolling volume ratio so build_signal_table can
    attach next-day volume information.

    Args:
        stock_df: DataFrame with columns: date, ticker, open, high, low, close, volume
        spy_df:   DataFrame with columns: date, spy_return

    Returns:
        DataFrame with columns: date, ticker, stock_return, spy_return,
                                 abnormal_return, volume_ratio
    """
    df = stock_df.copy().sort_values(["ticker", "date"])

    df["stock_return"] = df.groupby("ticker")["close"].pct_change()

    # Rolling 30-day average volume — ratio shows unusual trading activity
    df["rolling_volume"] = (
        df.groupby("ticker")["volume"]
        .transform(lambda s: s.rolling(30, min_periods=10).mean())
    )
    df["volume_ratio"] = df["volume"] / df["rolling_volume"]

    df = df.merge(spy_df, on="date", how="left")
    df["abnormal_return"] = df["stock_return"] - df["spy_return"]

    return df[["date", "ticker", "stock_return", "spy_return", "abnormal_return", "volume_ratio"]]


def build_signal_table(spike_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join spike days with next-day stock outcomes.

    spike_df must have a 'ticker' column (mapped externally from 'article').

    Args:
        spike_df:   Output of detect_spikes(), with an added 'ticker' column
        returns_df: Output of compute_abnormal_returns()

    Returns:
        DataFrame with columns:
            date, ticker, views, z_score, is_spike, spike_tier,
            next_day_abnormal_return, next_day_volume_ratio,
            next_2day_abnormal_return, next_3day_abnormal_return,
            next_5day_abnormal_return
    """
    # Compute next-day values by shifting within each ticker group
    ret = returns_df.copy().sort_values(["ticker", "date"])
    ret["next_day_abnormal_return"] = ret.groupby("ticker")["abnormal_return"].shift(-1)
    ret["next_day_volume_ratio"] = ret.groupby("ticker")["volume_ratio"].shift(-1)

    # Multi-lag cumulative abnormal returns (CAR): sum of daily abnormal returns t+1 to t+n
    for n in [2, 3, 5]:
        ret[f"next_{n}day_abnormal_return"] = sum(
            ret.groupby("ticker")["abnormal_return"].shift(-i) for i in range(1, n + 1)
        )

    lag_cols = [f"next_{n}day_abnormal_return" for n in [2, 3, 5]]
    merged = spike_df.merge(
        ret[["date", "ticker", "next_day_abnormal_return", "next_day_volume_ratio"] + lag_cols],
        on=["date", "ticker"],
        how="left",
    )

    # Build output column list — include z_score/spike_tier if detect_spikes added them
    optional = [c for c in ["z_score", "spike_tier"] if c in merged.columns]
    out_cols = (["date", "ticker", "views"] + optional +
                ["is_spike", "next_day_abnormal_return", "next_day_volume_ratio"] + lag_cols)

    return merged[out_cols]


if __name__ == "__main__":
    # Map: Wikipedia article title → stock ticker
    ARTICLE_TICKER = {
        "Apple Inc.": "AAPL",
        "Microsoft": "MSFT",
        "Tesla, Inc.": "TSLA",
    }

    START_WIKI = "20240101"
    END_WIKI = "20250301"
    START_STOCK = "2024-01-01"
    END_STOCK = "2025-03-01"

    articles = list(ARTICLE_TICKER.keys())
    tickers = list(ARTICLE_TICKER.values())

    print("Fetching Wikipedia pageviews...")
    views_df = get_pageviews_bulk(articles, START_WIKI, END_WIKI)

    print("\nDetecting spikes...")
    spike_df = detect_spikes(views_df)
    spike_df["ticker"] = spike_df["article"].map(ARTICLE_TICKER)

    n_spikes = spike_df["is_spike"].sum()
    print(f"  Found {n_spikes} spike days across {len(articles)} articles")

    print("\nFetching stock data...")
    stock_df = get_stock_data_bulk(tickers, START_STOCK, END_STOCK)

    print("\nFetching SPY benchmark...")
    spy_df = get_spy(START_STOCK, END_STOCK)

    print("\nComputing abnormal returns...")
    returns_df = compute_abnormal_returns(stock_df, spy_df)

    print("\nBuilding signal table...")
    signal_df = build_signal_table(spike_df, returns_df)

    # Show spike days with their next-day outcomes
    spikes_only = signal_df[signal_df["is_spike"]].dropna(subset=["next_day_abnormal_return"])
    spikes_only = spikes_only.sort_values("views", ascending=False)

    print(f"\nTop spike days (highest views) with next-day outcomes:")
    print(spikes_only[["date", "ticker", "views", "next_day_abnormal_return", "next_day_volume_ratio"]]
          .head(15)
          .to_string(index=False))

    print("\nSanity check — mean next-day abnormal return on spike vs. non-spike days:")
    summary = signal_df.dropna(subset=["next_day_abnormal_return"]).groupby("is_spike")["next_day_abnormal_return"].describe()
    print(summary)
