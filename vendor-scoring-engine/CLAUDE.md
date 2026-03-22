# Alt Data Vendor Scoring Engine

## Project overview
A reproducible evaluation framework for scoring alternative data vendors against six dimensions that matter to a discretionary L/S equity hedge fund. The engine ingests candidate datasets, runs them through a battery of statistical tests, and outputs composite scores with diagnostic visualizations via an interactive dashboard and REST API.

## Architecture
```
ingestion → evaluation pipeline (6 modules) → results store → dashboard + API
```

### Evaluation modules (src/modules/)
1. **coverage** — Universe fit, sector/cap bias, coverage over time
2. **signal_decay** — Information coefficient at varying lags, half-life estimation
3. **orthogonality** — Factor regression vs FF5 + momentum, spanning tests, residual IC
4. **backtest_integrity** — Point-in-time validation, backfill detection via changepoint analysis
5. **data_quality** — Missingness patterns, outlier frequency, delivery reliability, format consistency
6. **alpha_economics** — Cost vs marginal IC contribution, break-even AUM estimation

Each module must:
- Accept a standardized `VendorDataset` object (see src/ingestion/schema.py)
- Return a `ModuleResult` with: score (0-100), confidence, diagnostics dict, narrative string
- Be independently testable with synthetic data

## Tech stack
- **Python 3.11+** — core language
- **pandas / polars** — data manipulation (prefer polars for large datasets)
- **statsmodels / scipy** — statistical tests
- **scikit-learn** — ML utilities (not for modeling, for preprocessing + metrics)
- **SQLite** — results store (upgrade path to Postgres)
- **FastAPI** — REST API
- **Streamlit** — dashboard
- **pytest** — testing
- **ruff** — linting and formatting

## Coding standards
- Type hints on all function signatures
- Docstrings on all public functions (Google style)
- Each module in its own file under src/modules/
- Config via dataclasses, not dicts
- No print statements — use logging module
- Tests mirror src/ structure under tests/

## Common commands
```bash
# Run all tests
pytest tests/ -v

# Run a specific module's tests
pytest tests/modules/test_coverage.py -v

# Lint and format
ruff check src/ --fix
ruff format src/

# Run the dashboard locally
streamlit run src/dashboard/app.py

# Run the API server
uvicorn src.api.main:app --reload

# Generate sample data for testing
python -m src.ingestion.sample_generator
```

## Key data contracts
- **VendorDataset**: ticker (str), date (datetime), signal_value (float), metadata (dict)
- **Universe**: ticker (str), sector (str), market_cap (float), index_membership (list[str])
- **ModuleResult**: module_name (str), score (float 0-100), confidence (float 0-1), diagnostics (dict), narrative (str), plots (list[Figure])

## Reference data locations
- Fama-French factors: data/reference/ff5_factors.csv
- Sample universe (Russell 3000): data/reference/universe.csv
- Sample vendor dataset: data/sample/

## Development phases
- Phase 1: Ingestion + coverage module (CURRENT)
- Phase 2: Signal decay + orthogonality modules
- Phase 3: Dashboard + API
- Phase 4: Backtest integrity + data quality + alpha economics

## Important context
This project supports an Investment Data Science internship at a multi-strategy hedge fund.
The end users are discretionary equity L/S portfolio managers and central data science researchers.
Outputs must be interpretable by non-technical PMs — narrative summaries matter as much as scores.
