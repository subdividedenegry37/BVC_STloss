"""End-to-end driver for the calibration and regime-break pipeline.

Assumes per-contract 1-minute OHLCV Parquet files have already been
produced under ``data/parquet/`` (see ``scripts/data_prep/convert.py``).

Each stage is a standalone module under ``src/``; this driver sets
PYTHONPATH to include each stage's directory and invokes it with the
standard --es/--nq/--rty/--outdir CLI.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
FEATURES_DIR = SRC / "features"

ES = REPO_ROOT / "data" / "parquet" / "ohlcv_1m_ES.parquet"
NQ = REPO_ROOT / "data" / "parquet" / "ohlcv_1m_NQ.parquet"
RTY = REPO_ROOT / "data" / "parquet" / "ohlcv_1m_RTY.parquet"


def run_cmd(script: Path, outdir: Path, extra: list[str] | None = None) -> None:
    extra = extra or []
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(script.parent), str(FEATURES_DIR), existing])
    cmd = [sys.executable, str(script),
           "--es", str(ES), "--nq", str(NQ), "--rty", str(RTY),
           "--outdir", str(outdir)] + extra
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    out_causal = REPO_ROOT / "results" / "calibration_tables" / "15year_causal"
    out_causal.mkdir(parents=True, exist_ok=True)

    run_cmd(SRC / "calibration" / "phase1_descriptive.py", out_causal,
            extra=["--warmup", "80", "--gap", "15min"])
    run_cmd(SRC / "calibration" / "phase1_causal.py", out_causal,
            extra=["--warmup", "80", "--gap", "15min"])
    run_cmd(SRC / "calibration" / "yearly_blocks.py", out_causal,
            extra=["--warmup", "80", "--gap", "15min"])

    out_regime = REPO_ROOT / "results" / "regime_break"
    out_regime.mkdir(parents=True, exist_ok=True)
    run_cmd(SRC / "regime" / "rolling_nu.py", out_regime,
            extra=["--warmup", "80", "--gap", "15min"])

    print("\nCalibration and regime-break stages complete.")
    print("See README.md and docs/METHODOLOGY.md for downstream stages.")


if __name__ == "__main__":
    main()
