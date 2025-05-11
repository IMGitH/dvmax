# src/visualization/view_feature_table.py

import polars as pl
import matplotlib.pyplot as plt
import os

def visualize_feature_snapshot(file_path: str, output_path: str = None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")

    df = pl.read_parquet(file_path)
    if df.height == 0:
        raise ValueError("Feature table is empty")

    df = df.sort("date")
    latest = df.tail(1).drop("date")

    # Convert to pandas for compatibility with matplotlib
    latest_pd = latest.to_pandas().iloc[0]

    plt.figure(figsize=(12, 5))
    latest_pd.plot(kind="bar")
    plt.title("Latest Feature Snapshot")
    plt.ylabel("Value")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis="y")

    if not output_path:
        output_path = os.path.splitext(file_path)[0] + "_latest_snapshot.png"

    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Saved snapshot plot to {output_path}")


if __name__ == "__main__":
    visualize_feature_snapshot("./output/aapl_features.parquet")
