"""Schema definitions for vendor dataset ingestion.

Defines the standardized data contracts that all evaluation modules
expect as input. Any raw vendor data must be transformed into these
structures before evaluation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd


class AssetClass(Enum):
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    FX = "fx"
    CRYPTO = "crypto"


class Frequency(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    EVENT_DRIVEN = "event_driven"


@dataclass
class VendorMetadata:
    """Metadata about the data vendor and dataset."""

    vendor_name: str
    dataset_name: str
    description: str = ""
    asset_class: AssetClass = AssetClass.EQUITY
    frequency: Frequency = Frequency.DAILY
    start_date: datetime | None = None
    end_date: datetime | None = None
    cost_annual_usd: float | None = None
    delivery_method: str = ""  # "api", "sftp", "s3", "email"
    update_lag_days: int | None = None  # how many days after event is data available


@dataclass
class VendorDataset:
    """Standardized container for a vendor's signal data.

    The core contract: every vendor dataset gets normalized into this
    structure before any evaluation module touches it.

    Args:
        data: DataFrame with required columns [ticker, date, signal_value]
              and optional columns [sector, market_cap, raw_value, metadata_json]
        metadata: VendorMetadata with dataset-level information
        universe: DataFrame defining the target trading universe with columns
                  [ticker, sector, market_cap] and optional [index_membership]
    """

    data: pd.DataFrame
    metadata: VendorMetadata
    universe: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        self._validate_data()

    def _validate_data(self) -> None:
        """Validate required columns and types."""
        required_cols = {"ticker", "date", "signal_value"}
        missing = required_cols - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if not pd.api.types.is_datetime64_any_dtype(self.data["date"]):
            self.data["date"] = pd.to_datetime(self.data["date"])

        if not pd.api.types.is_numeric_dtype(self.data["signal_value"]):
            raise ValueError("signal_value must be numeric")

    @property
    def tickers(self) -> list[str]:
        return sorted(self.data["ticker"].unique().tolist())

    @property
    def date_range(self) -> tuple[datetime, datetime]:
        return self.data["date"].min(), self.data["date"].max()

    @property
    def n_observations(self) -> int:
        return len(self.data)

    @property
    def n_tickers(self) -> int:
        return self.data["ticker"].nunique()

    @property
    def n_dates(self) -> int:
        return self.data["date"].nunique()


@dataclass
class UniverseDefinition:
    """Defines the trading universe to evaluate coverage against.

    This is what the PM actually trades — the vendor data is only
    useful insofar as it covers this universe.
    """

    data: pd.DataFrame  # [ticker, sector, market_cap, index_membership]
    name: str = "custom"
    as_of_date: datetime | None = None

    def __post_init__(self) -> None:
        required = {"ticker", "sector", "market_cap"}
        missing = required - set(self.data.columns)
        if missing:
            raise ValueError(f"Universe missing columns: {missing}")

    @property
    def tickers(self) -> list[str]:
        return sorted(self.data["ticker"].unique().tolist())

    def sector_breakdown(self) -> dict[str, int]:
        """Count of tickers per sector."""
        return self.data.groupby("sector")["ticker"].nunique().to_dict()

    def cap_quintiles(self) -> pd.DataFrame:
        """Market cap quintile boundaries."""
        return self.data["market_cap"].describe(percentiles=[0.2, 0.4, 0.6, 0.8])
