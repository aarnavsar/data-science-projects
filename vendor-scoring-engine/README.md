# Alt Data Vendor Scoring Engine

A reproducible evaluation framework for scoring alternative data vendors across six dimensions that matter to discretionary L/S equity hedge funds.

## Quick Start

```bash
# Clone and setup
git clone <your-repo-url>
cd vendor-scoring-engine
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Generate sample data
python -m src.ingestion.sample_generator

# Run tests
pytest

# Launch dashboard (Phase 3)
streamlit run src/dashboard/app.py
```

## Evaluation Dimensions

| Module | What it measures | Key question |
|--------|-----------------|--------------|
| Coverage | Universe fit, sector/cap bias | Can I use this for the stocks I trade? |
| Signal Decay | IC over lag horizons | How quickly does the edge disappear? |
| Orthogonality | Factor regression residuals | Does this tell me something I don't already know? |
| Backtest Integrity | PIT validation, changepoints | Is the history real or backfilled? |
| Data Quality | Missingness, outliers, reliability | Can I depend on this in production? |
| Alpha Economics | Cost vs marginal IC | Does the math work at my AUM? |

## Project Structure

```
vendor-scoring-engine/
├── CLAUDE.md                    # Claude Code project context
├── .claude/commands/            # Custom Claude Code commands
├── src/
│   ├── ingestion/               # Data loading, schema validation
│   ├── modules/                 # The 6 evaluation modules
│   ├── scoring/                 # Result aggregation, composite scoring
│   ├── api/                     # FastAPI REST endpoints
│   └── dashboard/               # Streamlit interactive dashboard
├── tests/                       # Mirror of src/ structure
├── data/
│   ├── sample/                  # Synthetic test data
│   └── reference/               # Fama-French factors, universe definitions
├── notebooks/                   # Exploratory analysis
└── docs/                        # Architecture docs, module specs
```

## Claude Code Workflow

This project is designed for iterative development with Claude Code:

```bash
# Plan a new module
/plan-module orthogonality

# Build from the plan
/build-module orthogonality

# Run a full evaluation
/evaluate data/sample/vendor_good.csv
```
