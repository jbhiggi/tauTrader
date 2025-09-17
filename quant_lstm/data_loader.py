import pandas as pd
from pathlib import Path

def load_data(path: Path) -> pd.DataFrame:
    """
    Load OHLCV data from a CSV file with validation and ordering checks.

    Parameters
    ----------
    path : Path
        Full path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with validated OHLCV data, sorted by timestamp.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If required columns are missing or timestamps are not increasing.
    RuntimeError
        If the CSV cannot be read for other reasons.
    """
    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]

    # --- Load ---
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Could not find data file: {path}")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load CSV {path}: {e}")

    # --- Check columns ---
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing required columns: {missing}. "
                         f"Found columns: {list(df.columns)}")

    # --- Sort by time ---
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Time monotonicity check ---
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("❌ Timestamp column is not strictly increasing after sort.")

    # --- Log summary ---
    print(f"📂 OHLC data loaded successfully from {path}")
    print(f"   Shape: {df.shape}, "
          f"Time span: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

    return df


from pathlib import Path

def prepare_results_dirs(root: Path, basename: str):
    """
    Create a standardized results directory structure for an experiment run.

    This function builds a nested folder layout under ``results/<basename>/`` 
    inside the repository root. It ensures that the following subdirectories 
    exist, creating them if necessary:

    - ``results/<basename>/`` → top-level container for this run
    - ``results/<basename>/figures/`` → stores plots and visual diagnostics
    - ``results/<basename>/modified_data/`` → stores processed or transformed data

    Parameters
    ----------
    root : Path
        Repository root path (typically the project-level ROOT).
    basename : str
        Name of the experiment or run. Used to create the subfolder under ``results/``.

    Returns
    -------
    dict of str -> Path
        Dictionary with keys:
        - ``"results"`` → Path to the base results directory
        - ``"figures"`` → Path to the figures directory
        - ``"modified_data"`` → Path to the modified data directory
    """
    results_dir = root / "results" / basename
    figures_dir = results_dir / "figures"
    modifed_data_dir = results_dir / "modified_data"

    # Create directories if they don't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    modifed_data_dir.mkdir(parents=True, exist_ok=True)

    return {"results": results_dir, "figures": figures_dir, "modified_data": modifed_data_dir}
