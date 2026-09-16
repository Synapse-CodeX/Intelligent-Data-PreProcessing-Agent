from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ColumnMetadata:
    """Metadata describing a single dataset column."""

    name: str
    data_type: str


@dataclass
class DatasetMetadata:
    """Metadata describing an ingested dataset."""

    file_name: str
    file_type: str
    row_count: int
    column_count: int
    columns: list[ColumnMetadata]


def build_dataset_metadata(
    df: pd.DataFrame,
    file_path: Path,
) -> DatasetMetadata:
    """Build dataset metadata from a DataFrame and its source file."""
    columns = [
        ColumnMetadata(
            name=str(column),
            data_type=str(df[column].dtype),
        )
        for column in df.columns
    ]

    return DatasetMetadata(
        file_name=file_path.name,
        file_type=file_path.suffix.lower().lstrip("."),
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
    )
