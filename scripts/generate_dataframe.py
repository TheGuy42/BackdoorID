"""
generate_dataframe.py
---------------------
Collect all test_results.csv files from experiment log folders, enrich each
row with metadata extracted from the sibling JSON config files, and write
the merged DataFrame to a single CSV.

Usage
-----
    python scripts/generate_dataframe.py --log_dir logs_RQ1
    python scripts/generate_dataframe.py --log_dir logs_RQ1 --output results/rq1_results.csv
    python scripts/generate_dataframe.py --log_dir logs_RQ1 --output results/rq1_results.csv --verbose
    python scripts/generate_dataframe.py --log_dir logs --recursive
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.better_logging import get_logger

logger = get_logger(__name__, level=logging.INFO)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten(obj: object, prefix: str = "", sep: str = "/") -> dict:
    """Recursively flatten a nested dict/list into a flat dict with dotted keys."""
    items: dict = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}{sep}{k}" if prefix else str(k)
            items.update(_flatten(v, new_key, sep))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{prefix}{sep}{i}" if prefix else str(i)
            items.update(_flatten(v, new_key, sep))
    else:
        items[prefix] = obj
    return items


def load_json_configs(version_dir: Path) -> dict:
    """
    Load every *.json file in *version_dir* (excluding hparams.yaml, which is
    not JSON anyway) and return a single flat dict whose keys are prefixed with
    the stem of the source file, e.g. ``training_config.max_epochs``.
    """
    meta: dict = {}
    for json_path in sorted(version_dir.glob("*.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Could not read {json_path}: {exc}")
            continue

        prefix = json_path.stem          # e.g. "training_config"
        flat = _flatten(data, prefix=prefix)
        meta.update(flat)

    return meta


def process_version_dir(version_dir: Path, exp_folder: str) -> pd.DataFrame | None:
    """
    Read *test_results.csv* from *version_dir*, attach metadata columns,
    and return the enriched DataFrame.  Returns None if the CSV is missing.
    """
    csv_path = version_dir / "test_results.csv"
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning(f"Could not read {csv_path}: {exc}")
        return None

    # Build all metadata columns at once to avoid DataFrame fragmentation
    config_meta = load_json_configs(version_dir)
    meta_df = pd.DataFrame(
        {"exp_folder": exp_folder, "version": version_dir.name, **config_meta},
        index=df.index,
    )

    return pd.concat([meta_df, df], axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_results(log_dir: Path, verbose: bool = False, recursive: bool = False) -> pd.DataFrame:
    """
    Walk *log_dir*, find every ``version_*/test_results.csv``, and return
    the merged DataFrame.

    Expected folder layout::

        <log_dir>/
            <exp_folder>/          ← arbitrary experiment name
                version_0/
                    test_results.csv
                    *.json          ← config files
                version_1/         ← optional additional versions
                    ...
    """
    frames: list[pd.DataFrame] = []
    missing: list[Path] = []

    if recursive:
        # Find all test_results.csv files anywhere under log_dir
        csv_paths = sorted(log_dir.rglob("test_results.csv"))
        for csv_path in csv_paths:
            version_dir = csv_path.parent
            # If version_dir is 'version_N', its parent is the experiment folder.
            # Otherwise, the version_dir itself is the experiment folder.
            if version_dir.name.startswith("version_"):
                exp_dir = version_dir.parent
            else:
                exp_dir = version_dir

            # Use the relative path from log_dir to exp_dir as the experiment label.
            try:
                exp_folder = str(exp_dir.relative_to(log_dir))
            except ValueError:
                exp_folder = exp_dir.name

            if verbose:
                print(f"  Processing: {exp_folder}/{version_dir.name}")

            df = process_version_dir(version_dir, exp_folder)
            if df is not None:
                frames.append(df)
    else:
        # Each direct child of log_dir is an experiment folder
        exp_dirs = sorted(p for p in log_dir.iterdir() if p.is_dir())

        for exp_dir in exp_dirs:
            exp_folder = exp_dir.name

            # Find all version sub-directories
            version_dirs = sorted(p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("version_"))

            if not version_dirs:
                # Some experiments may not have a versioned sub-directory at all;
                # try reading directly from the exp_dir itself.
                version_dirs = [exp_dir]

            for version_dir in version_dirs:
                if verbose:
                    print(f"  Processing: {exp_folder}/{version_dir.name}")
                df = process_version_dir(version_dir, exp_folder)
                if df is not None:
                    frames.append(df)
                else:
                    missing.append(version_dir / "test_results.csv")

    if missing:
        logger.info(f"{len(missing)} version dir(s) had no test_results.csv:")
        for p in missing:
            logger.info(f"  {p}")

    if not frames:
        raise RuntimeError(f"No test_results.csv files found under '{log_dir}'.")

    merged = pd.concat(frames, ignore_index=True)
    logger.info(f"Merged {len(frames)} result file(s) → {len(merged):,} rows, {len(merged.columns)} columns.")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge all test_results.csv files from an experiments log folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        required=True,
        help="Root folder containing experiment sub-directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for the output CSV file. Defaults to <log_dir>/results.csv.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each processed experiment.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Treat log_dir as a root and find experiments nested at any level.",
    )
    args = parser.parse_args()

    log_dir: Path = args.log_dir.expanduser().resolve()
    if not log_dir.is_dir():
        logger.error(f"log_dir '{log_dir}' is not a directory.")
        sys.exit(1)

    output: Path = args.output if args.output is not None else log_dir / "results.csv"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning: {log_dir}")
    df = collect_results(log_dir, verbose=args.verbose, recursive=args.recursive)

    df.to_csv(output, index=False)
    logger.info(f"Saved → {output}")


if __name__ == "__main__":
    main()
