from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseTransformer(ABC):
    """Base interface for preprocessing transformations."""

    name: str

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseTransformer":
        """Fit the transformer using the provided dataset."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the provided dataset."""
        raise NotImplementedError

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the transformer and transform the dataset."""
        return self.fit(df).transform(df)

    def get_config(self) -> dict[str, Any]:
        """Return the transformer configuration."""
        return {
            "name": self.name,
        }
