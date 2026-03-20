import requests
import pandas as pd
from datetime import datetime


BASE_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
HEADERS = {"User-Agent": "Mozilla/5.0 (learning project, educational use)"}


def get_pageviews(article: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily Wikipedia pageviews for a single article.

    Args:
        article: Wikipedia article title (e.g. "Apple Inc.")
        start:   Start date as "YYYYMMDD"
        end:     End date as "YYYYMMDD"

    Returns:
        DataFrame with columns: date, article, views
    """
    # Replace spaces with underscores — Wikimedia API requires this
    article_slug = article.replace(" ", "_")

    url = f"{BASE_URL}/en.wikipedia/all-access/all-agents/{article_slug}/daily/{start}/{end}"

    response = requests.get(url, headers=HEADERS)

    if response.status_code == 404:
        print(f"  [not found] {article}")
        return pd.DataFrame(columns=["date", "article", "views"])

    response.raise_for_status()

    items = response.json()["items"]

    rows = []
    for item in items:
        rows.append({
            "date": pd.to_datetime(item["timestamp"], format="%Y%m%d00"),
            "article": article,
            "views": item["views"],
        })

    return pd.DataFrame(rows)


def get_pageviews_bulk(articles: list, start: str, end: str) -> pd.DataFrame:
    """
    Fetch pageviews for a list of articles and combine into one DataFrame.

    Args:
        articles: List of Wikipedia article titles
        start:    Start date as "YYYYMMDD"
        end:      End date as "YYYYMMDD"

    Returns:
        DataFrame with columns: date, article, views
    """
    dfs = []
    for i, article in enumerate(articles):
        print(f"  ({i+1}/{len(articles)}) {article}")
        df = get_pageviews(article, start, end)
        if not df.empty:
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


if __name__ == "__main__":
    # Quick test: fetch pageviews for a few companies over the last 30 days
    test_articles = ["Apple Inc.", "Microsoft", "Tesla, Inc."]
    start = "20250201"
    end = "20250301"

    print(f"Fetching pageviews from {start} to {end}...\n")
    df = get_pageviews_bulk(test_articles, start, end)

    print(f"\nFetched {len(df)} rows")
    print(df.head(10))
