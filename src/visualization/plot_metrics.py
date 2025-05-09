import polars as pl
import matplotlib.pyplot as plt
import os

def plot_metrics(df: pl.DataFrame, metric_cols: list[str], output_dir: str, title_prefix: str = ""):
    os.makedirs(output_dir, exist_ok=True)
    df = df.sort("date")

    for metric in metric_cols:
        if metric not in df.columns:
            print(f"[SKIP] {metric} not in dataframe")
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["date"].to_numpy(), df[metric].to_numpy(), marker='o')
        ax.set_title(f"{title_prefix} {metric}".strip())
        ax.set_xlabel("Date")
        ax.set_ylabel(metric)
        ax.grid(True)
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

    fetcher = StockFetcher()
    ticker = "AAPL"
    ratios = fetcher.fetch_ratios(ticker)
    output = f"./output/{ticker.lower()}_ratios"

    if ratios.height:
        plot_all_available_metrics(ratios, output, title_prefix=f"{ticker.upper()} Ratio")
    else:
        print("[WARN] No data available to plot.")
