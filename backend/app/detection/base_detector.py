from abc import ABC, abstractmethod

import pandas as pd

from app.schemas.issues import DataQualityIssue


class BaseDetector(ABC):
    """Base interface for all dataset quality detectors."""

    name: str

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> list[DataQualityIssue]:
        """Detect data quality issues in a dataset."""
        raise NotImplementedError
