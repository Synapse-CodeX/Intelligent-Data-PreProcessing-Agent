from pathlib import Path

import pandas as pd
import pytest
from app.core.exceptions import DatasetError, DatasetValidationError
from app.ingestion.csv_loader import load_csv
from app.ingestion.excel_loader import load_excel
from app.ingestion.json_loader import load_json
from app.ingestion.loader import load_dataset
from app.ingestion.validators import (
    validate_columns,
    validate_dataframe,
    validate_unique_columns,
)

SAMPLE_CSV = Path("data/sample/messy_customer_data.csv")


def test_csv_loader():
    df = load_csv(SAMPLE_CSV)

    assert not df.empty
    assert df.shape == (15, 7)


def test_load_dataset_returns_metadata():
    df, metadata = load_dataset(SAMPLE_CSV)

    assert df.shape == (15, 7)
    assert metadata.file_name == "messy_customer_data.csv"
    assert metadata.file_type == "csv"
    assert metadata.row_count == 15
    assert metadata.column_count == 7


def test_load_dataset_missing_file():
    missing_file = Path("data/sample/does_not_exist.csv")

    with pytest.raises(DatasetError, match="does not exist"):
        load_dataset(missing_file)


def test_validate_dataframe_rejects_empty_dataset():
    import pandas as pd

    with pytest.raises(DatasetValidationError, match="Dataset is empty"):
        validate_dataframe(pd.DataFrame())


def test_validate_columns_rejects_empty_column_name():
    df = pd.DataFrame(
        [[1, 2]],
        columns=["name", "   "],
    )

    with pytest.raises(
        DatasetValidationError,
        match="empty column name",
    ):
        validate_columns(df)


def test_validate_unique_columns_rejects_duplicates():
    df = pd.DataFrame(
        [[1, 25]],
        columns=["customer_id", "customer_id"],
    )

    with pytest.raises(
        DatasetValidationError,
        match="duplicate column names",
    ):
        validate_unique_columns(df)


def test_excel_loader():
    file_path = Path("data/sample/test_customers.xlsx")

    source_df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "name": ["John", "Alice", "Bob"],
            "age": [25, 32, 28],
        }
    )

    try:
        source_df.to_excel(file_path, index=False)

        df = load_excel(file_path)

        assert df.shape == (3, 3)
        assert list(df.columns) == ["customer_id", "name", "age"]
    finally:
        if file_path.exists():
            file_path.unlink()


def test_json_loader():
    file_path = Path("data/sample/test_customers.json")

    source_df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "name": ["John", "Alice", "Bob"],
            "age": [25, 32, 28],
        }
    )

    try:
        source_df.to_json(file_path, orient="records")

        df = load_json(file_path)

        assert df.shape == (3, 3)
        assert list(df.columns) == ["customer_id", "name", "age"]
    finally:
        if file_path.exists():
            file_path.unlink()


def test_load_dataset_excel():
    file_path = Path("data/sample/test_customers.xlsx")

    source_df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "name": ["John", "Alice", "Bob"],
            "age": [25, 32, 28],
        }
    )

    try:
        source_df.to_excel(file_path, index=False)

        df, metadata = load_dataset(file_path)

        assert df.shape == (3, 3)
        assert metadata.file_name == "test_customers.xlsx"
        assert metadata.file_type == "xlsx"
        assert metadata.row_count == 3
        assert metadata.column_count == 3
    finally:
        if file_path.exists():
            file_path.unlink()


def test_load_dataset_json():
    file_path = Path("data/sample/test_customers.json")

    source_df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "name": ["John", "Alice", "Bob"],
            "age": [25, 32, 28],
        }
    )

    try:
        source_df.to_json(file_path, orient="records")

        df, metadata = load_dataset(file_path)

        assert df.shape == (3, 3)
        assert metadata.file_name == "test_customers.json"
        assert metadata.file_type == "json"
        assert metadata.row_count == 3
        assert metadata.column_count == 3
    finally:
        if file_path.exists():
            file_path.unlink()
