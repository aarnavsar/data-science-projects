Run a full vendor evaluation on the specified dataset: $ARGUMENTS

1. Read the dataset file and validate schema against VendorDataset contract
2. Load reference data (universe, factors) from data/reference/
3. Run each available module in src/modules/ sequentially
4. Aggregate results into composite score using src/scoring/aggregator.py
5. Generate the evaluation report with all diagnostic plots
6. Store results in the SQLite database
7. Print a summary with composite score, per-module breakdown, and key findings
