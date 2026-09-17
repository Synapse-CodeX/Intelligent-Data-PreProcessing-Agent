import pandas as pd
from app.services.validation_service import ValidationService
from app.validation.comparison import DatasetComparison
from app.validation.dataset_validator import DatasetValidator
from app.validation.leakage_validator import LeakageValidator
from app.validation.preprocessing_validator import (
    PreprocessingValidator,
)
from app.validation.schema_validator import SchemaValidator


def test_dataset_validator():
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["Kolkata", "Delhi", "Mumbai"],
        }
    )

    result = DatasetValidator().validate(df)

    assert result["passed"] is True
    assert result["checks"]["dataset_not_empty"] is True
    assert result["checks"]["column_names_unique"] is True


def test_dataset_validator_rejects_empty_dataset():
    df = pd.DataFrame()

    result = DatasetValidator().validate(df)

    assert result["passed"] is False
    assert result["checks"]["dataset_not_empty"] is False


def test_schema_validator():
    original_df = pd.DataFrame(
        {
            "age": [20, 30],
            "city": ["Kolkata", "Delhi"],
        }
    )

    processed_df = pd.DataFrame(
        {
            "age": [20, 30],
            "city_Kolkata": [1, 0],
        }
    )

    result = SchemaValidator().validate(
        original_df,
        processed_df,
    )

    assert result["passed"] is True
    assert "city" in result["removed_columns"]
    assert "city_Kolkata" in result["added_columns"]


def test_preprocessing_validator():
    original_df = pd.DataFrame(
        {
            "age": [20, 30, None],
        }
    )

    processed_df = pd.DataFrame(
        {
            "age": [20, 30, 25],
        }
    )

    result = PreprocessingValidator().validate(
        original_df,
        processed_df,
    )

    assert result["passed"] is True
    assert result["checks"]["row_count_preserved"] is True
    assert result["checks"]["no_missing_values"] is True
    assert result["checks"]["no_infinite_values"] is True


def test_preprocessing_validator_detects_missing_values():
    original_df = pd.DataFrame(
        {
            "age": [20, 30, None],
        }
    )

    processed_df = pd.DataFrame(
        {
            "age": [20, None, 25],
        }
    )

    result = PreprocessingValidator().validate(
        original_df,
        processed_df,
    )

    assert result["passed"] is False
    assert result["checks"]["no_missing_values"] is False


def test_leakage_validator():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [1, 2, 3],
        }
    )

    result = LeakageValidator().validate(
        df,
        target_column="target",
    )

    assert result["passed"] is False
    assert len(result["issues"]) == 1
    assert result["issues"][0]["column"] == "feature"


def test_leakage_validator_without_target():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    result = LeakageValidator().validate(df)

    assert result["passed"] is True
    assert result["checks"]["target_column_provided"] is False


def test_dataset_comparison():
    original_df = pd.DataFrame(
        {
            "age": [20, 30, None],
            "city": ["Kolkata", "Delhi", "Mumbai"],
        }
    )

    processed_df = pd.DataFrame(
        {
            "age": [20, 30, 25],
            "city": ["Kolkata", "Delhi", "Mumbai"],
        }
    )

    result = DatasetComparison().compare(
        original_df,
        processed_df,
    )

    assert result["original"]["rows"] == 3
    assert result["processed"]["rows"] == 3
    assert result["original"]["missing_values"] == 1
    assert result["processed"]["missing_values"] == 0
    assert result["changes"]["missing_values"] == -1


def test_validation_service():
    original_df = pd.DataFrame(
        {
            "age": [20, 30, None],
            "city": ["Kolkata", "Delhi", "Mumbai"],
        }
    )

    processed_df = pd.DataFrame(
        {
            "age": [20, 30, 25],
            "city": ["Kolkata", "Delhi", "Mumbai"],
        }
    )

    service = ValidationService()

    result = service.validate(
        original_df,
        processed_df,
    )

    assert result["passed"] is True
    assert result["preprocessing"]["passed"] is True
    assert result["comparison"]["original"]["missing_values"] == 1
    assert result["comparison"]["processed"]["missing_values"] == 0
