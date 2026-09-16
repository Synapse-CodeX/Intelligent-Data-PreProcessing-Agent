from pathlib import Path

import pandas as pd


def load_json(file_path: Path) -> pd.DataFrame:
    """Load a JSON file into a pandas DataFrame."""
    return pd.read_json(file_path)
