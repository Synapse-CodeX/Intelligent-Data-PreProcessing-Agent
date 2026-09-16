from pathlib import Path

import pandas as pd
from app.ingestion.loader import load_dataset
from app.profiling.column_profiler import profile_columns
from app.profiling.correlations import calculate_correlations
from app.profiling.data_types import infer_data_types, infer_semantic_types
from app.profiling.distributions import analyze_distributions
from app.profiling.profiler import profile_dataset
from app.profiling.statistics import calculate_statistics

DATASET_PATH = Path("data/sample/messy_customer_data.csv")


def load_test_dataset():
    """Load the sample dataset used by profiling tests."""
    df, _ = load_dataset(DATASET_PATH)
    return df


def test_profile_dataset():
    df = load_test_dataset()

    profile = profile_dataset(df)

    assert profile["row_count"] == 15
    assert profile["column_count"] == 7
    assert profile["columns"] == list(df.columns)


def test_infer_data_types():
    df = load_test_dataset()

    data_types = infer_data_types(df)

    assert set(data_types) == set(df.columns)
    assert data_types["age"] == str(df["age"].dtype)


def test_profile_columns():
    df = load_test_dataset()

    profiles = profile_columns(df)

    assert profiles["income"]["missing_count"] == 2
    assert profiles["city"]["missing_count"] == 1
    assert profiles["gender"]["missing_count"] == 1
    assert profiles["name"]["unique_count"] == 13


def test_calculate_statistics():
    df = load_test_dataset()

    statistics = calculate_statistics(df)

    assert "age" in statistics
    assert "income" in statistics
    assert "purchase_amount" in statistics
    assert "city" not in statistics

    assert "mean" in statistics["income"]
    assert "median" in statistics["income"]
    assert "q1" in statistics["income"]
    assert "q3" in statistics["income"]


def test_analyze_distributions():
    df = load_test_dataset()

    distributions = analyze_distributions(df)

    assert "age" in distributions
    assert "income" in distributions
    assert "purchase_amount" in distributions
    assert "skewness" in distributions["income"]
    assert "kurtosis" in distributions["income"]


def test_calculate_correlations():
    df = load_test_dataset()

    correlations = calculate_correlations(df)

    assert "age" not in correlations
    assert "income" in correlations
    assert "purchase_amount" in correlations
    assert correlations["income"]["income"] == 1.0


def test_profile_dataset_contains_all_sections():
    df = load_test_dataset()

    profile = profile_dataset(df)

    expected_sections = {
        "row_count",
        "column_count",
        "columns",
        "data_types",
        "column_profiles",
        "statistics",
        "distributions",
        "correlations",
    }

    assert expected_sections.issubset(profile.keys())


def test_infer_semantic_types():
    df = pd.DataFrame(
        {
            "age": ["20", "25", "30", "35"],
            "city": ["Kolkata", "Delhi", "Mumbai", "Kolkata"],
        }
    )

    semantic_types = infer_semantic_types(df)

    assert semantic_types["age"] == "numeric"
    assert semantic_types["city"] == "categorical"


def test_infer_semantic_types_with_dirty_numeric_column():
    df = pd.DataFrame(
        {
            "age": ["20", "25", "abc", "30", "35"],
        }
    )

    semantic_types = infer_semantic_types(df)

    assert semantic_types["age"] == "numeric"


def test_infer_semantic_types_datetime():
    df = pd.DataFrame(
        {
            "date": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
            ],
        }
    )

    semantic_types = infer_semantic_types(df)

    assert semantic_types["date"] == "datetime"
