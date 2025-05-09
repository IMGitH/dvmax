import polars as pl
import matplotlib.pyplot as plt
import os
import numpy as np


def plot_metrics(df: pl.DataFrame, metric_cols: list[str], output_dir: str, title_prefix: str = ""):
    os.makedirs(output_dir, exist_ok=True)
    df = df.sort("date")

    for metric in metric_cols:
        if metric not in df.columns:
            print(f"[SKIP] {metric} not in dataframe")
            continue

        values = df[metric].to_numpy()
        dates = df["date"].to_numpy()

        mean = np.mean(values)
        std = np.std(values)
        upper_bound = mean + 2 * std
        lower_bound = mean - 2 * std

        anomalies = (values > upper_bound) | (values < lower_bound)

        _, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dates, values, marker='o', label=f"{metric}", color='blue')

        if anomalies.any():
            ax.scatter(dates[anomalies], values[anomalies], color='red', label=f"{metric} Anomaly")

        ax.axhline(upper_bound, color='gray', linestyle='--', linewidth=0.8)
        ax.axhline(lower_bound, color='gray', linestyle='--', linewidth=0.8)

        ax.set_title(f"{title_prefix} {metric}".strip())
        ax.set_xlabel("Date")
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        output_path = os.path.join(output_dir, f"{metric}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"[INFO] Saved: {output_path}")


def plot_all_available_metrics(df: pl.DataFrame, output_dir: str, title_prefix: str = ""):
    metric_cols = [col for col in df.columns if col != "date"]
    plot_metrics(df, metric_cols, output_dir, title_prefix)


if __name__ == "__main__":
    from src.dataprep.fetcher import StockFetcher
    from src.dataprep.features import FeatureEngineer

    fetcher = StockFetcher()
    engineer = FeatureEngineer()

    ticker = "AAPL"
    ratios = fetcher.fetch_ratios(ticker)
    prices = fetcher.fetch_prices(ticker)

    if prices.height:
        ret_6m = engineer.compute_6m_return(prices)
        vol = engineer.compute_volatility(prices)
        print(f"[INFO] {ticker} - 6M Return: {ret_6m:.2%}, Volatility: {vol:.2%}")

    output = f"./output/{ticker.lower()}_ratios"
    if ratios.height:
        plot_all_available_metrics(ratios, output, title_prefix=f"{ticker.upper()} Ratio")
    else:
        print("[WARN] No ratio data available to plot.")
