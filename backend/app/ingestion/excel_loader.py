from pathlib import Path

import pandas as pd


def load_excel(file_path: Path) -> pd.DataFrame:
    """Load an Excel file into a pandas DataFrame."""
    return pd.read_excel(file_path)
