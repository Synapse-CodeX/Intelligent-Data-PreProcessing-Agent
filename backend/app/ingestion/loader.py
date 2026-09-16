from pathlib import Path

import pandas as pd

from app.core.exceptions import DatasetError
from app.ingestion.csv_loader import load_csv
from app.ingestion.excel_loader import load_excel
from app.ingestion.json_loader import load_json
from app.ingestion.schema import DatasetMetadata, build_dataset_metadata
from app.ingestion.validators import (
    validate_columns,
    validate_dataframe,
    validate_unique_columns,
)


def load_dataset(
    file_path: Path,
) -> tuple[pd.DataFrame, DatasetMetadata]:
    """Load, validate, and describe a supported dataset."""
    if not file_path.exists():
        raise DatasetError(f"Dataset file does not exist: {file_path}")

    file_type = file_path.suffix.lower()

    if file_type == ".csv":
        df = load_csv(file_path)
    elif file_type in {".xlsx", ".xls"}:
        df = load_excel(file_path)
    elif file_type == ".json":
        df = load_json(file_path)
    else:
        raise DatasetError(f"Unsupported dataset format: {file_type or 'unknown'}")

    validate_dataframe(df)
    validate_columns(df)
    validate_unique_columns(df)

    metadata = build_dataset_metadata(df, file_path)

    return df, metadata
