# Wikipedia Alternative Data — Signal Study

Does a spike in Wikipedia page views for an S&P 500 company predict abnormal stock returns or trading volume in the days that follow?

This project builds a full alternative data pipeline — from raw API pulls to event-study statistics — and answers that question with ~2 years of daily data across ~500 companies.

---

## Findings

### Volume signal: strong and significant
Wikipedia spikes reliably predict elevated next-day trading volume (p ≈ 0, t-stat > 10). Mean next-day volume ratio on spike days is ~1.20x vs. ~1.00x on non-spike days. The effect is consistent across sectors and spike magnitudes.

### Return signal: weak at 1-day, stronger over 2–5 days
A naive 1-day lag shows no significant return effect (p = 0.11). But the signal builds over time:

| Holding period | p-value | Significant? |
|----------------|---------|--------------|
| 1-day CAR      | 0.109   |              |
| 2-day CAR      | 0.036   | *            |
| 3-day CAR      | 0.012   | *            |
| 5-day CAR      | 0.005   | ***          |

This suggests Wikipedia attention is followed by a slow drift — consistent with gradual information diffusion or delayed institutional awareness, not immediate price impact.

### Extreme spikes carry the real signal
Splitting spikes by z-score tier reveals the signal is concentrated in rare, high-magnitude events:

| Tier | Threshold | n | Next-day return | Next-day volume |
|------|-----------|---|-----------------|-----------------|
| Weak | 2–3σ | 8,084 | +0.02% | 1.16x |
| Medium | 3–5σ | 3,534 | +0.04% | 1.23x |
| Strong | 5σ+ | 286 | **+0.22%** | **1.59x** |

Strong-tier spikes (e.g. CrowdStrike outage, UnitedHealth CEO news) show ~10x larger return signal and 60% elevated volume vs. baseline.

### Sector breakdown
Financials is the only sector with a significant 1-day return signal (p = 0.01). Other sectors show no significant return effect, though the volume signal is broad-based.

### Strategy metrics (long-on-spike, naive)
A simple strategy that longs every spike day and sells the next day is not investable as-is:
- Hit rate: 49.5% (coin flip)
- Annualized Sharpe: 0.18
- Max drawdown: -5.7 (cumulative CAR)

The natural extension — restricting to strong-tier spikes only — was not tested but is the obvious next step.

---

## Project Structure

```
wiki-alternative-data/
├── fetch_data.py          # Pull raw data from Wikimedia API + Yahoo Finance
├── process_data.py        # Build processed outputs from raw data
├── src/
│   ├── signals.py         # Spike detection, abnormal returns, signal table
│   ├── wiki_pageviews.py  # Wikimedia REST API client (checkpointed bulk fetch)
│   ├── stock_data.py      # yfinance wrapper for OHLCV + SPY
│   └── sp500.py           # S&P 500 constituent list with Wikipedia article mapping
├── notebooks/
│   ├── 01_fetch_data.ipynb      # Data overview and quality checks
│   ├── 02_spike_detection.ipynb # Spike detection methodology and EDA
│   └── 03_analysis.ipynb        # Event study, sector breakdown, multi-lag, strategy metrics
├── data/
│   ├── raw/               # wiki_pageviews.csv, stock_data.csv, spy.csv, sp500.csv
│   └── processed/         # spikes.csv, returns.csv, signal_table.csv
└── results/               # Saved charts (PNG)
```

---

## Data

- **Wikipedia pageviews:** Wikimedia REST API, daily granularity, ~498 S&P 500 articles, ~2 years
- **Stock prices:** Yahoo Finance via `yfinance`, daily OHLCV for ~503 tickers
- **Benchmark:** SPY daily returns used to compute abnormal returns
- **Sample:** ~370K article-days, ~11,900 spike events with matched stock outcomes

---

## Methodology

### Spike detection
For each Wikipedia article, compute a 30-day rolling mean and standard deviation of daily views (minimum 10 periods). A day is flagged as a spike if:

```
z_score = (views - rolling_mean) / rolling_std > 2
```

Spikes are further classified into tiers: weak (2–3σ), medium (3–5σ), strong (5σ+).

### Abnormal returns
Daily stock return minus SPY return on the same day. A 30-day rolling average volume is used to compute a volume ratio (day's volume / rolling mean volume).

### Signal table
Each article-day is joined to next-day stock outcomes (abnormal return, volume ratio). Multi-lag cumulative abnormal returns (CAR) are computed as the sum of daily abnormal returns over 2, 3, and 5 trading days following the spike.

### Statistical test
Welch's two-sample t-test comparing spike days vs. non-spike days on each outcome variable. Sector-level tests require at least 10 spike days per sector.

---

## Reproducing the Analysis

```bash
# 1. Activate the environment
source /Users/ucla/myenv/bin/activate

# 2. Fetch raw data (Wikimedia rate-limits ~400 articles per run; run twice to get all ~500)
cd wiki-alternative-data
python fetch_data.py   # first run: ~400 articles
python fetch_data.py   # second run: resumes from checkpoint, gets the rest

# 3. Build processed outputs
python process_data.py

# 4. Open notebooks in order with the myenv kernel
#    01_fetch_data.ipynb → 02_spike_detection.ipynb → 03_analysis.ipynb
```

---

## Limitations

- **Survivorship bias:** S&P 500 constituents are as of the data pull date. Companies that were added or removed during the sample period are handled inconsistently.
- **Short sample:** ~2 years is too short for reliable out-of-sample validation or robust multiple-testing correction.
- **No earnings alignment:** Spikes around earnings announcements are not separated from organic attention spikes, which likely dilutes the signal.
- **Simple benchmark:** SPY is used as the sole benchmark. Beta-adjusted or Fama-French abnormal returns would be more rigorous.
- **Article mapping:** Wikipedia article titles are matched to tickers manually/heuristically; some mismatches are possible.

---

## Stack

Python 3 · `pandas` · `numpy` · `scipy` · `yfinance` · `requests` · `matplotlib` · `seaborn` · Jupyter
