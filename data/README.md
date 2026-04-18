# Data

This directory is the target for the underlying OHLCV inputs. The raw
data is not redistributed with the repository; users must acquire it
from the upstream vendor.

## Source

- **Vendor:** Databento
- **Dataset:** CME Globex MDP 3.0 (`GLBX.MDP3`)
- **Schema:** `ohlcv-1m` (1-minute OHLCV bars, aggregated from trades)
- **Symbols:** `ES`, `NQ`, `RTY` root symbols with raw-symbol filter of
  the form `{ROOT}{H,M,U,Z}{digit}` (the continuous front-month is
  constructed downstream by volume-based rolls).
- **Window:** 2010-01-01 through 2025-12-31 (inclusive) for the
  15-year analyses; any sub-period for smaller runs.
- **Fields:** `ts_event`, `instrument_id`, `open`, `high`, `low`,
  `close`, `volume`, and the symbol metadata fields
  (`raw_symbol`, `expiration`, etc.).

## Layout

Place the vendor's CSV batch under `data/raw/`:

```
data/
  raw/                       # raw Databento CSVs (user-supplied)
    glbx-mdp3-*.ohlcv-1m.csv
  parquet/                   # per-contract Parquet produced by convert.py
    ohlcv_1m_ES.parquet
    ohlcv_1m_NQ.parquet
    ohlcv_1m_RTY.parquet
```

## Conversion

`scripts/data_prep/convert.py` reads every `*.csv` under `data/raw/`,
partitions by root symbol (`ES`/`NQ`/`RTY`), enforces the standard
dtype schema, and writes one Parquet file per contract into
`data/parquet/`.

```bash
python scripts/data_prep/convert.py
```

The remaining utilities in `scripts/data_prep/` provide audit and
validation helpers: `analyze_csvs.py`, `audit_data.py`, `audit_v2.py`,
`inspect_csv.py`, `validate_outputs.py`, `validate_v3.py`.

## Intermediate feature caches

The Stage-1 pipeline writes per-instrument feature dataframes as
`phase2_features_*.pkl` under `results/calibration_tables/`,
`results/regime_break/`, and
`results/stage1_volatility/physics_features/`. These pickles are
intermediate caches, not primary results: they are regenerated
deterministically by `scripts/run_full_pipeline.py` from the raw
Databento bars and the code in `src/`. To keep the repository within
GitHub's file-size limits and to avoid shipping large redundant
artifacts, all `*.pkl` files are excluded from version control via
`.gitignore`. The small CSV and Parquet summaries that the paper
actually cites remain tracked.

## Licensing

Databento data is redistributed under the subscriber's agreement with
Databento. Only the derived, aggregated result artifacts under
`results/` are published with this repository; the underlying OHLCV
bars are not.
