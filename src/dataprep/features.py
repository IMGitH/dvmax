import polars as pl
import datetime

class FeatureEngineer:
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir

    def compute_6m_return(self, df: pl.DataFrame, as_of_date: datetime.date | None = None) -> float:
        """Compute 6-month price return from a general DataFrame."""
        df = ensure_date_column(df, "date").sort("date")

        if "close" not in df.columns:
            raise ValueError("Expected a 'close' column in the DataFrame")

        if as_of_date is None:
            as_of_date = datetime.date.today()

        # Approximate 6 months as 180 days
        six_months_ago = as_of_date - datetime.timedelta(days=6*30)

        price_now = find_nearest_price(df, as_of_date)
        price_six_months_ago = find_nearest_price(df, six_months_ago)

        # Return percentage change over 6 months
        return (price_now - price_six_months_ago) / price_six_months_ago

    def compute_volatility(self, price_df: pl.DataFrame) -> float:
        """Compute annualized volatility based on daily returns."""
        price_df = ensure_date_column(price_df, "date").sort("date")

        if "close" not in price_df.columns:
            raise ValueError("Expected a 'close' column in the DataFrame")

        # Compute daily returns
        returns = price_df.select((pl.col("close") / pl.col("close").shift(1) - 1).alias("daily_return")).drop_nulls()
        std_dev = returns["daily_return"].std()

        if std_dev is None:
            return 0.0

        # Annualize daily standard deviation: multiply by sqrt(252) because ~252 trading days/year
        return std_dev * (252 ** 0.5)

def ensure_date_column(df: pl.DataFrame, column_name: str = "date") -> pl.DataFrame:
    """Ensure a column is properly parsed as Polars Date type."""
    if df[column_name].dtype != pl.Date:
        return df.with_columns(
            pl.col(column_name).str.strptime(pl.Date, "%Y-%m-%d")
        )
    return df

def find_nearest_price(df: pl.DataFrame, target_date: datetime.date) -> float:
    """Find closing price closest to target date."""
    df = df.with_columns(
        # Calculate absolute difference between each row date and target date
        (pl.col("date") - pl.lit(target_date)).abs().alias("date_diff")
    ).sort("date_diff")
    return df[0, "close"]
