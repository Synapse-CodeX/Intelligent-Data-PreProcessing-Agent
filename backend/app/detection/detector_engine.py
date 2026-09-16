import pandas as pd

from app.detection.base_detector import BaseDetector
from app.detection.categorical import CategoricalConsistencyDetector
from app.detection.constant_features import ConstantFeatureDetector
from app.detection.datetime import DateTimeDetector
from app.detection.duplicates import DuplicateDetector
from app.detection.missing_values import MissingValueDetector
from app.detection.outliers import OutlierDetector
from app.detection.skewness import SkewnessDetector
from app.schemas.issues import DataQualityIssue


class DetectorEngine:
    """Orchestrate dataset quality detectors."""

    def __init__(
        self,
        detectors: list[BaseDetector] | None = None,
    ) -> None:
        self.detectors = detectors or [
            MissingValueDetector(),
            DuplicateDetector(),
            OutlierDetector(),
            CategoricalConsistencyDetector(),
            ConstantFeatureDetector(),
            SkewnessDetector(),
            DateTimeDetector(),
        ]

    def detect(self, df: pd.DataFrame) -> list[DataQualityIssue]:
        """Run all registered detectors on a dataset."""
        issues = []

        for detector in self.detectors:
            issues.extend(detector.detect(df))

        return issues
