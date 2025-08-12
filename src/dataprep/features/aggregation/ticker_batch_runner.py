import os
import time
from types import SimpleNamespace
from datetime import date, datetime, timezone
from typing import List
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import sys

import polars as pl
from tqdm import tqdm

from src.dataprep.fetcher._fmp_client import fmp_get
from src.dataprep.fetcher.ticker_data_sources import fetch_all_per_ticker
from src.dataprep.features.aggregation.ticker_row_builder import build_feature_table_from_inputs
from src.dataprep.features.aggregation.validate_dynamic_row import validate_dynamic_row
from src.dataprep.constants import EXPECTED_COLUMNS


ACCEPT_STATUSES = {"ok", "flagged"}  # keep flagged rows, audit them


# ---- live progress helper ----
_progress_hist: list[tuple[float, int]] = []  # (timestamp, processed) in last ~2m

def _update_progress_live(status_dir: str, totals: dict, counts: dict, running=None, note: str | None = None) -> None:
    """
    totals: {"total_tasks": int, "tickers": int, "dates": int}
    counts: {"processed": int, "failed": int, "flagged": int}
    running: {"ticker": str, "as_of": str} | None
    """
    now = time.time()
    _progress_hist.append((now, int(counts.get("processed", 0))))
    # keep only last 120s window
    cutoff = now - 120
    while _progress_hist and _progress_hist[0][0] < cutoff:
        _progress_hist.pop(0)

    eta_iso = None
    if len(_progress_hist) >= 2:
        dt = _progress_hist[-1][0] - _progress_hist[0][0]
        ditems = _progress_hist[-1][1] - _progress_hist[0][1]
        rate = (ditems / dt) if dt > 0 else 0.0
        remaining = max(int(totals.get("total_tasks", 0)) - int(counts.get("processed", 0)), 0)
        if rate > 0:
            secs = int(remaining / rate)
            eta_iso = datetime.fromtimestamp(now + secs, tz=timezone.utc).replace(microsecond=0).isoformat()

    denom = max(int(totals.get("total_tasks", 0)), 1)
    percent = round(100 * int(counts.get("processed", 0)) / denom, 2)

    os.makedirs(status_dir, exist_ok=True)
    path = os.path.join(status_dir, "progress.json")

    # preserve started_at from previous writes
    started_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                prev = json.load(f)
                started_at = prev.get("started_at") or started_at
    except Exception as _e:
        print(f"[WARN] progress read failed: {_e}")

    payload = {
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "totals": totals,
        "counts": counts,
        "percent": percent,
        "eta_utc": eta_iso,
        "running": running or {},
        "note": note or "",
    }
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as _e:
        print(f"[WARN] progress write failed: {_e}")


# ---------------- Run stats ----------------
@dataclass
class RunStats:
    ok: int = 0
    skipped: int = 0
    flagged: int = 0   # soft validations triggered
    failed: int = 0    # code/runtime failures
    changed_tickers: int = 0

    def update_from_lines(self, lines: list[str]) -> None:
        for ln in lines:
            if ln.startswith("[OK] "): self.ok += 1
            elif ln.startswith("[SKIP] "): self.skipped += 1
            elif ln.startswith("[FLAGGED] "): self.flagged += 1
            elif ln.startswith("[FAIL] "): self.failed += 1


def _maybe_preflight_fmp():
    """Run FMP preflight only if explicitly enabled via env."""
    if os.environ.get("FMP_PREFLIGHT", "0") in ("0", "", "false", "False"):
        print("⏭️  Skipping FMP preflight (FMP_PREFLIGHT=0).")
        return
    try:
        _ = fmp_get("/api/v3/ratios/AAPL", {"limit": 1})
        print("✅ FMP auth OK")
    except Exception as e:
        # Keep behavior for CLI runs: fail fast
        raise SystemExit(f"❌ FMP check failed: {e}")


# === Configuration (simplified) ===
START_DATE = date(2021, 12, 31)  # <-- adjust your backtesting start
END_DATE = date.today().replace(month=12, day=31)  # e.g. last full year
FREQ = "1Y"

TICKERS_FILE = "features_data/tickers/us_tickers_subset_limited.txt"
SLEEP_BETWEEN_CALLS = 1.0
MAX_RETRIES = 3

# Modes:
# - "append":    skip existing rows for each ticker/date; append only new ones
# - "overwrite": ignore existing per-ticker file; rebuild entirely
# - "skip":      if a per-ticker file exists, skip that ticker entirely
OVERWRITE_MODE = os.environ.get("OVERWRITE_MODE", "append").lower()
VALID_OVERWRITE_MODES = {"append", "overwrite", "skip"}
if OVERWRITE_MODE not in VALID_OVERWRITE_MODES:
    raise ValueError(f"Invalid OVERWRITE_MODE: '{OVERWRITE_MODE}'. Must be one of {VALID_OVERWRITE_MODES}")

# Force merge regardless of mtimes; e.g. FORCE_MERGE=1 python script.py
FORCE_MERGE = os.environ.get("FORCE_MERGE", "0") not in ("0", "", "false", "False")
# STRICT now only considers 'failed'; 'flagged' never fails the run
STRICT = os.environ.get("STRICT", "0") not in ("0", "", "false", "False")

OUTPUT_DIR = os.path.join("features_data", "tickers_history")
AUDIT_DIR  = os.path.join("features_data", "_audit")
STATIC_DIR = os.path.join("features_data", "tickers_static")
STATUS_DIR = os.path.join("features_data", "status")


# === Utilities ===
def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(AUDIT_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(STATUS_DIR, exist_ok=True)


def get_dates_between(start: date, end: date, freq: str = "1Y") -> List[date]:
    current = start
    dates = []
    while current <= end:
        dates.append(current)
        current = current.replace(year=current.year + 1)
    return dates


def load_tickers(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    with open(file_path) as f:
        return [line.strip().upper() for line in f if line.strip()]


def get_parquet_path(ticker: str) -> str:
    return os.path.join(OUTPUT_DIR, f"{ticker}.parquet")


def is_numeric_dtype(dtype: pl.DataType) -> bool:
    return isinstance(dtype, (
        pl.Float32, pl.Float64,
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64
    ))


def fill_missing_columns(df: pl.DataFrame, all_columns: list[str]) -> pl.DataFrame:
    height = df.height
    to_add = []
    for col in all_columns:
        if col not in df.columns:
            # keep the SAME number of rows; for empty df this adds 0-length series
            to_add.append(pl.Series(name=col, values=[None] * height))
    if to_add:
        df = df.with_columns(to_add)
    return df.select(all_columns)


def cast_and_round_numeric(df: pl.DataFrame) -> pl.DataFrame:
    cols = []
    for col in df.columns:
        dtype = df[col].dtype
        if is_numeric_dtype(dtype):
            cols.append(pl.col(col).round(2).cast(pl.Float32).alias(col))
        else:
            cols.append(pl.col(col))
    return df.select(cols)


def save_static_row(static_df: pl.DataFrame):
    static_path = os.path.join(STATIC_DIR, "static_ticker_info.parquet")
    os.makedirs(os.path.dirname(static_path), exist_ok=True)

    if os.path.exists(static_path):
        existing = pl.read_parquet(static_path)
    else:
        existing = pl.DataFrame()

    # Align columns
    all_columns = sorted(set(existing.columns) | set(static_df.columns))
    existing = fill_missing_columns(existing, all_columns)
    static_df = fill_missing_columns(static_df, all_columns)

    # Resolve a common dtype per column:
    # - prefer non-Null
    # - if either is Utf8 -> Utf8
    # - if both numeric -> Int32 (compact)  [use Float32 if you prefer]
    # - else fall back to the non-Null one
    def _common_dtype(de: pl.DataType, dn: pl.DataType) -> pl.DataType:
        if de == pl.Null and dn != pl.Null:
            return dn
        if dn == pl.Null and de != pl.Null:
            return de
        if de == pl.Utf8 or dn == pl.Utf8:
            return pl.Utf8
        if is_numeric_dtype(de) and is_numeric_dtype(dn):
            return pl.Int32
        if de != pl.Null:
            return de
        return dn

    target_schema: dict[str, pl.DataType] = {}
    for col in all_columns:
        de = existing[col].dtype
        dn = static_df[col].dtype
        if de != dn:
            target_schema[col] = _common_dtype(de, dn)

    if target_schema:
        existing = existing.cast(target_schema)
        static_df = static_df.cast(target_schema)

    # Normalize one-hots: cast to Int8 and fill missing as 0
    ohe_cols = [c for c in all_columns if c.startswith("sector_") or c.startswith("country_")]
    if ohe_cols:
        existing = existing.with_columns([pl.col(c).cast(pl.Int8).fill_null(0) for c in ohe_cols])
        static_df = static_df.with_columns([pl.col(c).cast(pl.Int8).fill_null(0) for c in ohe_cols])

    combined = pl.concat([existing, static_df], how="vertical").unique(subset=["ticker"])
    combined.write_parquet(static_path, compression="zstd")
    print(f"💾 Static info saved (total: {combined.height} rows) → {static_path}")


def save_or_append(df: pl.DataFrame, ticker: str, merge_with_existing: bool = True) -> bool:
    """
    Writes/merges the per-ticker parquet. Returns True if the file content changed.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"[BUG] df is not a Polars DataFrame for {ticker}. Got: {type(df)}")
    if "as_of" not in df.columns:
        raise ValueError(f"[BUG] 'as_of' missing in df for {ticker}. Got columns: {df.columns}")

    path = get_parquet_path(ticker)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prev_height = None

    if merge_with_existing and os.path.exists(path):
        df_existing = pl.read_parquet(path)

        # Ensure validation cols exist on existing file as Utf8 empty strings
        for c in ("validation_status", "violations"):
            if c not in df_existing.columns:
                df_existing = df_existing.with_columns(pl.lit("").cast(pl.Utf8).alias(c))

        # Align columns
        all_columns = sorted(set(df.columns) | set(df_existing.columns))
        df = fill_missing_columns(df, all_columns)
        df_existing = fill_missing_columns(df_existing, all_columns)

        # Normalize numerics (keeps strings intact)
        df = cast_and_round_numeric(df)
        df_existing = cast_and_round_numeric(df_existing)

        # Harmonize dtypes
        cast_schema = {}
        for col in all_columns:
            de = df_existing[col].dtype
            dn = df[col].dtype
            if de != dn:
                if is_numeric_dtype(de) and is_numeric_dtype(dn):
                    target = pl.Float32
                else:
                    if de == pl.Null and dn != pl.Null:
                        target = dn
                    elif dn == pl.Null and de != pl.Null:
                        target = de
                    elif de == pl.Utf8 or dn == pl.Utf8:
                        target = pl.Utf8
                    elif de == pl.Boolean or dn == pl.Boolean:
                        target = pl.Boolean
                    else:
                        target = de
                cast_schema[col] = target
        if cast_schema:
            df = df.cast(cast_schema)
            df_existing = df_existing.cast(cast_schema)

        # Merge
        combined = pl.concat([df_existing, df], how="vertical").unique(subset=["as_of"])

        # FINAL GUARANTEES for validation cols:
        #  - Utf8 dtype
        #  - Preserve any non-empty values; default to "" only if truly null
        if "validation_status" in combined.columns:
            combined = combined.with_columns(
                pl.coalesce([pl.col("validation_status").cast(pl.Utf8), pl.lit("")]).alias("validation_status")
            )
        else:
            combined = combined.with_columns(pl.lit("").alias("validation_status"))

        if "violations" in combined.columns:
            combined = combined.with_columns(
                pl.coalesce([pl.col("violations").cast(pl.Utf8), pl.lit("")]).alias("violations")
            )
        else:
            combined = combined.with_columns(pl.lit("").alias("violations"))

        prev_height = df_existing.height
        df = combined
    else:
        df = cast_and_round_numeric(df)


    df = df.sort("as_of")
    new_height = df.height

    # If we merged with existing and height changed -> changed
    if (prev_height is not None and new_height != prev_height):
        df.write_parquet(path, compression="zstd")
        print(f"💾 Saved: {ticker} ({df.height} rows, compressed) → {path}")
        return True

    # If we did NOT merge (overwrite mode) and the file already exists,
    # treat overwrite as "changed" (content is different, even if bytes match)
    if not merge_with_existing and os.path.exists(path):
        df.write_parquet(path, compression="zstd")
        print(f"💾 Saved (overwrite): {ticker} ({df.height} rows, compressed) → {path}")
        return True

    # Otherwise: do the temp write + byte-compare fallback
    tmp_path = path + ".tmp"
    df.write_parquet(tmp_path, compression="zstd")
    same_bytes = False
    try:
        same_bytes = os.path.getsize(tmp_path) == os.path.getsize(path)
    except FileNotFoundError:
        same_bytes = False
    os.replace(tmp_path, path)
    print(f"💾 Saved: {ticker} ({df.height} rows, compressed) → {path}")
    return not same_bytes


def fetch_and_build_features(ticker: str, as_of: date) -> tuple[pl.DataFrame, pl.DataFrame, str | None]:
    inputs = fetch_all_per_ticker(ticker, div_lookback_years=5, other_lookback_years=5)

    if not has_enough_price_data(inputs, as_of):
        raise ValueError("Not enough price data available for required historical features")

    dynamic_df, static_df = build_feature_table_from_inputs(ticker, inputs, as_of)

    if not isinstance(dynamic_df, pl.DataFrame):
        raise TypeError(f"[BUG] dynamic_df is not a Polars DataFrame for {ticker}@{as_of} — got {type(dynamic_df)}")
    if not isinstance(static_df, pl.DataFrame):
        raise TypeError(f"[BUG] static_df is not a Polars DataFrame for {ticker}@{as_of} — got {type(static_df)}")
    if dynamic_df.is_empty():
        raise ValueError(f"[WARN] build_feature_table_from_inputs returned empty dynamic_df for {ticker}@{as_of}")

    # Try to extract sector (optional)
    sector = None
    if "sector" in static_df.columns:
        try:
            sector = static_df["sector"].to_list()[0]
        except Exception:
            sector = None

    dynamic_df = dynamic_df.with_columns([
        pl.lit(ticker).alias("ticker"),
        pl.lit(as_of).cast(pl.Date).alias("as_of")
    ])
    dynamic_df = dynamic_df.select(["ticker", "as_of"] + [c for c in dynamic_df.columns if c not in ("ticker", "as_of")])

    return dynamic_df, static_df, sector


def _write_flagged_audit(ticker: str, as_of, errors: list, row_df: pl.DataFrame) -> None:
    Path(AUDIT_DIR).mkdir(parents=True, exist_ok=True)
    fname = f"{ticker}_{as_of}.txt"  # test expects this pattern
    audit_fp = Path(AUDIT_DIR) / fname
    with open(audit_fp, "w") as f:
        f.write("\n".join(errors) if errors else "")

def _align_schemas(df1: pl.DataFrame, df2: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Ensure both DataFrames have the same columns
    all_cols = list(dict.fromkeys(df1.columns + df2.columns))  # preserve order
    for col in all_cols:
        if col not in df1.columns:
            df1 = df1.with_columns(pl.lit(None).alias(col))
        if col not in df2.columns:
            df2 = df2.with_columns(pl.lit(None).alias(col))
    # Reorder columns
    return df1.select(all_cols), df2.select(all_cols)


def generate_features_for_ticker(
    ticker: str,
    dates: list[date],
    on_progress=None
):
    logs: list[str] = []
    stats = RunStats()  # ok, skipped, flagged, failed, changed_tickers

    out_fp = Path(OUTPUT_DIR) / f"{ticker}.parquet"
    if out_fp.exists():
        df_existing = pl.read_parquet(out_fp)
        # Ensure legacy files have validation cols as empty strings
        for c in ("validation_status", "violations"):
            if c not in df_existing.columns:
                df_existing = df_existing.with_columns(pl.lit("").cast(pl.Utf8).alias(c))
    else:
        df_existing = pl.DataFrame()

    existing_dates = set(df_existing["as_of"].to_list()) if not df_existing.is_empty() else set()
    keep_df = df_existing
    changed = False
    saved_static = False
    for as_of in dates:
        if as_of in existing_dates:
            stats.skipped += 1
            logs.append(f"[SKIP] {ticker} {as_of} exists — skipping")
            if on_progress:
                on_progress(status="processed", flagged=False, ticker=ticker, as_of=as_of)
            continue

        try:
            dyn_df, static_df, sector = fetch_and_build_features(ticker, as_of)

            if not saved_static:
                # ensure ticker col exists (it does) and dtypes are strings/ints
                save_static_row(static_df)
                saved_static = True

            status, errors, dyn_df = validate_dynamic_row(
                dyn_df, ticker,
                prev_df=keep_df if not keep_df.is_empty() else None,
                sector=sector
            )

            # Ensure validation columns exist on the new row
            dyn_df = dyn_df.with_columns([
                pl.lit(status).cast(pl.Utf8).alias("validation_status"),
                pl.lit(";".join(errors)).cast(pl.Utf8).alias("violations"),
            ])

            if status in ACCEPT_STATUSES:
                if status == "flagged":
                    stats.flagged += 1
                    _write_flagged_audit(ticker, as_of, errors, dyn_df)
                else:
                    stats.ok += 1

                # Align schemas and append
                keep_df, dyn_df = _align_schemas(keep_df, dyn_df)
                keep_df = pl.concat([keep_df, dyn_df], how="vertical_relaxed")
                existing_dates.add(as_of)
                changed = True
                logs.append(f"[OK] {ticker} {as_of} status={status} added")
                if on_progress:
                    on_progress(status="processed", flagged=(status == "flagged"), ticker=ticker, as_of=as_of)
            else:
                logs.append(f"[SKIP] {ticker} {as_of} dropped (status={status}) errors={errors}")
                stats.skipped += 1
                if on_progress:
                    on_progress(status="processed", flagged=False, ticker=ticker, as_of=as_of)

        except Exception as e:
            stats.failed += 1
            err = f"[FAIL] {ticker} {as_of}: {type(e).__name__}: {e}"
            logs.append(err)
            if on_progress:
                on_progress(status="failed", flagged=False, ticker=ticker, as_of=as_of)
            # keep going to next date

    if changed:
        keep_df = keep_df.unique(subset=["as_of"], keep="last").sort("as_of")
        keep_df.write_parquet(out_fp, compression="zstd")
        stats.changed_tickers += 1

    return logs, changed, stats


def has_enough_price_data(inputs: dict, as_of: date, required_days: int = 260) -> bool:
    if "prices" not in inputs:
        return False
    df = inputs["prices"]
    return df.filter(pl.col("date") <= pl.lit(as_of)).height >= required_days


def merge_all_feature_vectors(force_merge: bool = False):
    paths = sorted(Path(OUTPUT_DIR).glob("*.parquet"))
    if not paths:
        raise RuntimeError("No feature vector files found.")

    merged_file = os.path.join(OUTPUT_DIR, "features_all_tickers_timeseries.parquet")

    def _parts_newer_than_merged() -> bool:
        if not os.path.exists(merged_file):
            return True
        merged_mtime = os.path.getmtime(merged_file)
        latest_part_mtime = max(os.path.getmtime(p) for p in paths)
        return latest_part_mtime > merged_mtime

    if not (force_merge or _parts_newer_than_merged()):
        print("⏩ Skipped merging – merged file is up to date.")
        return

    # Build superset of columns
    all_columns = set()
    for path in paths:
        df = pl.read_parquet(path)
        all_columns.update(df.columns)
    all_columns = sorted(all_columns)

    dfs = []
    for path in paths:
        df = pl.read_parquet(path)
        df = fill_missing_columns(df, all_columns)
        # Enforce consistent dtypes (floats for numerics)
        schema = {}
        for col in df.columns:
            dtype = df[col].dtype
            if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
                schema[col] = pl.Float64
            elif dtype in [pl.Utf8, pl.Boolean, pl.Float64, pl.Date]:
                schema[col] = dtype
            else:
                schema[col] = pl.Float64
        df = df.cast(schema)
        dfs.append(df)

    merged_df = pl.concat(dfs, how="vertical").sort(["ticker", "as_of"])
    merged_df.write_parquet(merged_file)
    print(f"✅ Merged {len(paths)} files into {merged_file}")


def _write_status_files(stats: RunStats):
    os.makedirs(STATUS_DIR, exist_ok=True)

    # processed.json (ledger)
    from glob import glob
    from pathlib import Path as _Path
    done = { _Path(p).stem: "ok" for p in glob(os.path.join(OUTPUT_DIR, "*.parquet")) }
    _Path(os.path.join(STATUS_DIR, "processed.json")).write_text(json.dumps(done, indent=2))

    # last_run.json (summary)
    _Path(os.path.join(STATUS_DIR, "last_run.json")).write_text(json.dumps(asdict(stats), indent=2))


def main():
    _maybe_preflight_fmp()
    ensure_output_dir()

    tickers = load_tickers(TICKERS_FILE)
    all_dates = get_dates_between(START_DATE, END_DATE)

    print(f"🟢 Generating features for {len(tickers)} tickers × {len(all_dates)} dates...")

    any_changed = False
    agg = RunStats()

    totals = {"total_tasks": len(tickers) * len(all_dates), "tickers": len(tickers), "dates": len(all_dates)}
    counts = {"processed": 0, "failed": 0, "flagged": 0}

    def _on_progress(status: str, flagged: bool, ticker, as_of):
        if status == "processed":
            counts["processed"] += 1
            if flagged:
                counts["flagged"] += 1
        elif status == "failed":
            counts["failed"] += 1
        _update_progress_live(
            STATUS_DIR,
            totals=totals,
            counts=counts,
            running={"ticker": str(ticker), "as_of": as_of.isoformat()},
        )

    # seed progress at 0%
    _update_progress_live(STATUS_DIR, totals, counts, note="starting")

    for ticker in tqdm(tickers, desc="Processing tickers"):
        logs, changed, tstats = generate_features_for_ticker(ticker, all_dates, on_progress=_on_progress)
        any_changed = any_changed or changed
        agg.ok += tstats.ok
        agg.skipped += tstats.skipped
        agg.flagged += tstats.flagged
        agg.failed += tstats.failed
        agg.changed_tickers += tstats.changed_tickers

        for ln in logs:
            tqdm.write(ln)
        time.sleep(SLEEP_BETWEEN_CALLS)

    merge_all_feature_vectors(force_merge=FORCE_MERGE or any_changed)
    write_static_ohe_projection()
    _write_status_files(agg)

    print(
        f"🏁 Summary: ok={agg.ok}, skipped={agg.skipped}, flagged={agg.flagged}, "
        f"failed={agg.failed}, changed_tickers={agg.changed_tickers}"
    )
    print(f"All dates generated: {[d.isoformat() for d in get_dates_between(START_DATE, END_DATE)]}")

    sys.exit(1 if agg.failed > 0 else 0)


def write_static_ohe_projection():
    src = os.path.join(STATIC_DIR, "static_ticker_info.parquet")
    dst = os.path.join(STATIC_DIR, "static_ohe.parquet")
    if not os.path.exists(src):
        return
    df = pl.read_parquet(src)
    ohe_cols = [c for c in df.columns if c.startswith("sector_") or c.startswith("country_")]
    if not ohe_cols:
        return
    out = df.select(["ticker"] + ohe_cols).with_columns(
        [pl.col(c).cast(pl.Float32).fill_null(0.0) for c in ohe_cols]
    )
    out.write_parquet(dst, compression="zstd")
    print(f"💾 Wrote OHE projection → {dst}")

if __name__ == "__main__":
    main()
