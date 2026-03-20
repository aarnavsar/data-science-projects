import yfinance as yf
import pandas as pd


def get_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for a single ticker via yfinance.

    Args:
        ticker: Stock ticker (e.g. "AAPL")
        start:  Start date as "YYYY-MM-DD"
        end:    End date as "YYYY-MM-DD"

    Returns:
        DataFrame with columns: date, ticker, open, high, low, close, volume
    """
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if raw.empty:
        print(f"  [no data] {ticker}")
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    df = df.reset_index()
    df["ticker"] = ticker
    df["date"] = pd.to_datetime(df["date"])

    return df[["date", "ticker", "open", "high", "low", "close", "volume"]]


def get_stock_data_bulk(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for a list of tickers.

    Args:
        tickers: List of stock tickers
        start:   Start date as "YYYY-MM-DD"
        end:     End date as "YYYY-MM-DD"

    Returns:
        DataFrame with columns: date, ticker, open, high, low, close, volume
    """
    dfs = []
    for i, ticker in enumerate(tickers):
        print(f"  ({i+1}/{len(tickers)}) {ticker}")
        df = get_stock_data(ticker, start, end)
        if not df.empty:
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def get_spy(start: str, end: str) -> pd.DataFrame:
    """
    Fetch SPY (S&P 500 ETF) daily returns — used as the market benchmark
    when computing abnormal returns.
    """
    df = get_stock_data("SPY", start, end)
    df["spy_return"] = df["close"].pct_change()
    return df[["date", "spy_return"]]


if __name__ == "__main__":
    test_tickers = ["AAPL", "MSFT", "TSLA"]
    start = "2025-02-01"
    end = "2025-03-01"

    print(f"Fetching stock data from {start} to {end}...\n")
    df = get_stock_data_bulk(test_tickers, start, end)

    print(f"\nFetched {len(df)} rows")
    print(df.head(10))

    print("\nFetching SPY benchmark...")
    spy = get_spy(start, end)
    print(spy.head())
