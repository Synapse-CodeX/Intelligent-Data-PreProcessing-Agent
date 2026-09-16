import pandas as pd

from app.core.exceptions import DatasetValidationError


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that a DataFrame exists and contains data."""
    if df is None:
        raise DatasetValidationError("Dataset is missing.")

    if df.empty:
        raise DatasetValidationError("Dataset is empty.")


def validate_columns(df: pd.DataFrame) -> None:
    """Validate that the DataFrame contains valid column names."""
    if len(df.columns) == 0:
        raise DatasetValidationError("Dataset contains no columns.")

    for column in df.columns:
        if not str(column).strip():
            raise DatasetValidationError("Dataset contains an empty column name.")


def validate_unique_columns(df: pd.DataFrame) -> None:
    """Validate that all DataFrame column names are unique."""
    if df.columns.duplicated().any():
        duplicated_columns = df.columns[df.columns.duplicated()].tolist()
        raise DatasetValidationError(
            f"Dataset contains duplicate column names: {duplicated_columns}"
        )
