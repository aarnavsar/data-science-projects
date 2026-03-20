import os
import time
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

    for attempt in range(3):
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 429:
            wait = 15
            for remaining in range(wait, 0, -1):
                print(f"  [rate limited] retrying in {remaining}s...  ", end="\r", flush=True)
                time.sleep(1)
            print()  # newline after countdown
            continue
        break

    if response.status_code == 429:
        print(f"  [skipped — still rate limited after retries] {article}")
        return pd.DataFrame(columns=["date", "article", "views"])

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


def get_pageviews_bulk(articles: list, start: str, end: str, checkpoint_path: str = None, batch_limit: int = None) -> pd.DataFrame:
    """
    Fetch pageviews for a list of articles and combine into one DataFrame.
    Supports checkpointing: saves progress to checkpoint_path as it goes,
    and skips already-fetched articles on resume.

    Args:
        articles:         List of Wikipedia article titles
        start:            Start date as "YYYYMMDD"
        end:              End date as "YYYYMMDD"
        checkpoint_path:  Optional path to a CSV file for incremental saves
        batch_limit:      Stop after fetching this many new articles (for session splitting).
                          Resume next run from checkpoint — already-fetched articles are skipped.

    Returns:
        DataFrame with columns: date, article, views
    """
    # Load already-fetched articles from checkpoint if it exists
    fetched_articles = set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        existing = pd.read_csv(checkpoint_path)
        fetched_articles = set(existing["article"].unique())
        print(f"  Resuming from checkpoint: {len(fetched_articles)} articles already fetched")

    # Load already-skipped (404) articles so we don't retry them
    skipped_path = checkpoint_path.replace(".csv", "_skipped.csv") if checkpoint_path else None
    known_skipped = set()
    if skipped_path and os.path.exists(skipped_path):
        known_skipped = set(pd.read_csv(skipped_path)["article"].tolist())
        print(f"  Skipping {len(known_skipped)} known-404 articles")

    remaining = [a for a in articles if a not in fetched_articles and a not in known_skipped]
    if batch_limit:
        remaining = remaining[:batch_limit]
        print(f"  Batch limit: fetching up to {batch_limit} articles this run")
    print(f"  {len(remaining)} articles to fetch ({len(fetched_articles)} already done, {len(known_skipped)} known 404s)\n")

    newly_skipped = []
    for i, article in enumerate(remaining):
        print(f"  ({len(fetched_articles)+i+1}/{len(articles)}) {article}")
        df = get_pageviews(article, start, end)

        if df.empty:
            newly_skipped.append(article)
            # Record 404s so they're not retried next run
            if skipped_path:
                pd.DataFrame({"article": [article]}).to_csv(
                    skipped_path, mode="a", header=not os.path.exists(skipped_path), index=False
                )
        else:
            if checkpoint_path:
                df.to_csv(checkpoint_path, mode="a", header=not os.path.exists(checkpoint_path), index=False)

        time.sleep(0.2)

    if newly_skipped:
        print(f"\n  Skipped {len(newly_skipped)} articles (rate limited or not found): {newly_skipped[:5]}{'...' if len(newly_skipped) > 5 else ''}")

    if batch_limit and len(remaining) == batch_limit:
        total_done = len(fetched_articles) + len(remaining)
        print(f"\n  Batch complete. {total_done}/{len(articles)} articles done total.")
        print(f"  Run again to continue from article {total_done + 1}.")

    # Return full dataset from checkpoint file
    if checkpoint_path and os.path.exists(checkpoint_path):
        return pd.read_csv(checkpoint_path, parse_dates=["date"])

    return pd.DataFrame()


if __name__ == "__main__":
    # Quick test: fetch pageviews for a few companies over the last 30 days
    test_articles = ["Apple Inc.", "Microsoft", "Tesla, Inc."]
    start = "20250201"
    end = "20250301"

    print(f"Fetching pageviews from {start} to {end}...\n")
    df = get_pageviews_bulk(test_articles, start, end)

    print(f"\nFetched {len(df)} rows")
    print(df.head(10))
