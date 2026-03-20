# Wikipedia Alternative Data Project

## Goal
Investigate whether spikes in Wikipedia page views for S&P 500 companies predict short-term changes in stock price, volume, or investor interest (1-day lag focus).

## Hypothesis
Abnormal Wikipedia traffic for a company → elevated market attention → measurable next-day effect on stock price movement or trading volume.

## Data Sources
- **Wikipedia pageviews:** Wikimedia REST API (`https://wikimedia.org/api/rest_v1/metrics/pageviews/`) — daily granularity, per article
- **Stock data:** Yahoo Finance via `yfinance` Python library — OHLCV data for S&P 500 tickers
- **S&P 500 constituents:** Wikipedia list or a static CSV

## Key Questions
1. Does a pageview spike (e.g., >2 std devs above rolling mean) predict next-day abnormal return?
2. Does it predict next-day volume spike?
3. Which sectors or company sizes show the strongest signal?
4. Is the signal stronger around earnings, news events, or random days?

## Project Structure (planned)
```
wiki-alternative-data/
  data/          # raw and processed data
  notebooks/     # EDA and analysis
  src/           # reusable data fetching and signal logic
  results/       # charts, tables, summary stats
```

## Stack
- Python: `pandas`, `numpy`, `yfinance`, `requests`, `matplotlib`/`seaborn`
- Notebooks: Jupyter

## Methodology (rough)
1. Pull daily Wikipedia pageviews for each S&P 500 company's Wikipedia article
2. Compute a rolling baseline (e.g., 30-day rolling mean/std) and flag spike days
3. Pull corresponding stock OHLCV data
4. Compute next-day abnormal return (vs. SPY as benchmark) and next-day volume ratio
5. Run event study / t-test to check if spike days have statistically different outcomes
6. Visualize and summarize findings

## Current Status & Next Steps
1. **Run `fetch_data.py`** — fetches all raw data into `data/raw/`. Wikimedia rate-limits around article 400, so run it twice: first run gets ~400 articles, second run resumes from checkpoint and gets the rest. Use the venv: `source /Users/ucla/myenv/bin/activate && python fetch_data.py`
2. Run `process_data.py` — builds `data/processed/` files needed by the notebooks
3. Open notebooks in order: `01_fetch_data`, `02_spike_detection`, `03_analysis`

## Notes
- This is a learning project — prioritize clarity and working code over optimization
- Use Claude Code for the full workflow: data fetching, analysis, visualization
