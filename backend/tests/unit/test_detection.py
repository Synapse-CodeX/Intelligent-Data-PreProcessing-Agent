import pandas as pd
from app.detection.categorical import CategoricalConsistencyDetector
from app.detection.constant_features import ConstantFeatureDetector
from app.detection.detector_engine import DetectorEngine
from app.detection.duplicates import DuplicateDetector
from app.detection.missing_values import MissingValueDetector
from app.detection.outliers import OutlierDetector
from app.detection.skewness import SkewnessDetector


def test_missing_value_detector():
    df = pd.DataFrame(
        {
            "age": [20, 25, None, 30],
            "city": ["Kolkata", "Delhi", "Mumbai", "Kolkata"],
        }
    )

    issues = MissingValueDetector().detect(df)

    assert len(issues) == 1
    assert issues[0].issue_type == "missing_values"
    assert issues[0].column == "age"
    assert issues[0].affected_count == 1


def test_duplicate_detector():
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Alice"],
            "age": [25, 30, 25],
        }
    )

    issues = DuplicateDetector().detect(df)

    assert len(issues) == 1
    assert issues[0].issue_type == "duplicate_rows"
    assert issues[0].affected_count == 1


def test_duplicate_detector_without_duplicates():
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
        }
    )

    issues = DuplicateDetector().detect(df)

    assert issues == []


def test_outlier_detector():
    df = pd.DataFrame(
        {
            "value": [10, 11, 12, 13, 100],
        }
    )

    issues = OutlierDetector().detect(df)

    assert len(issues) == 1
    assert issues[0].issue_type == "outliers"
    assert issues[0].column == "value"
    assert issues[0].affected_count == 1
    assert issues[0].metadata["method"] == "IQR"


def test_categorical_consistency_detector():
    df = pd.DataFrame(
        {
            "city": ["Mumbai", "mumbai", "Delhi", "Kolkata"],
        }
    )

    issues = CategoricalConsistencyDetector().detect(df)

    assert len(issues) == 1
    assert issues[0].issue_type == "categorical_inconsistency"
    assert issues[0].column == "city"
    assert "mumbai" in issues[0].metadata["inconsistent_groups"]


def test_constant_feature_detector():
    df = pd.DataFrame(
        {
            "country": ["India", "India", "India"],
            "age": [20, 25, 30],
        }
    )

    issues = ConstantFeatureDetector().detect(df)

    assert len(issues) == 1
    assert issues[0].issue_type == "constant_feature"
    assert issues[0].column == "country"


def test_skewness_detector():
    df = pd.DataFrame(
        {
            "value": [1, 1, 1, 1, 20],
        }
    )

    issues = SkewnessDetector().detect(df)

    assert len(issues) == 1
    assert issues[0].issue_type == "skewness"
    assert issues[0].column == "value"


def test_detector_engine():
    df = pd.DataFrame(
        {
            "age": [20, 25, None, 30],
            "city": ["Mumbai", "mumbai", "Delhi", "Kolkata"],
            "country": ["India", "India", "India", "India"],
        }
    )

    issues = DetectorEngine().detect(df)

    issue_types = {issue.issue_type for issue in issues}

    assert "missing_values" in issue_types
    assert "categorical_inconsistency" in issue_types
    assert "constant_feature" in issue_types


def test_outlier_detector_detects_numeric_like_object_column():
    df = pd.DataFrame(
        {
            "age": [
                "20",
                "21",
                "22",
                "23",
                "24",
                "25",
                "26",
                "27",
                "28",
                "150",
                "abc",
            ]
        }
    )

    issues = OutlierDetector().detect(df)

    outlier_issues = [issue for issue in issues if issue.issue_type == "outliers"]

    assert len(outlier_issues) == 1
    assert outlier_issues[0].column == "age"
    assert outlier_issues[0].affected_count == 1
