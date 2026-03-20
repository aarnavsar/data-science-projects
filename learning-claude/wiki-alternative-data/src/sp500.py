import requests
import pandas as pd
from bs4 import BeautifulSoup


def get_sp500() -> pd.DataFrame:
    """
    Scrape the S&P 500 constituent list from Wikipedia using BeautifulSoup.
    Returns a DataFrame with columns: ticker, company, sector, wiki_article
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (learning project, educational use)"}

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    # The S&P 500 table has id="constituents"
    table = soup.find("table", {"id": "constituents"})

    rows = []
    for tr in table.tbody.find_all("tr"):
        cols = tr.find_all("td")
        if not cols:
            continue  # skip header row

        ticker = cols[0].text.strip()
        company = cols[1].text.strip()
        sector = cols[2].text.strip()

        # Grab the Wikipedia article title from the link in the company cell
        link = cols[1].find("a")
        wiki_article = link["title"] if link and link.get("title") else company

        rows.append({
            "ticker": ticker.replace(".", "-"),  # BRK.B -> BRK-B for yfinance
            "company": company,
            "sector": sector,
            "wiki_article": wiki_article,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = get_sp500()
    print(f"Loaded {len(df)} S&P 500 companies")
    print(df.head(10))
