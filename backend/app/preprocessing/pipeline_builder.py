from typing import Any

import pandas as pd

from app.preprocessing.base_transformer import BaseTransformer
from app.preprocessing.encoding import CategoricalEncoder
from app.preprocessing.imputation import MissingValueImputer
from app.preprocessing.scaling import NumericalScaler
from app.schemas.preprocessing import PreprocessingConfig


class PreprocessingPipeline:
    """Sequential preprocessing pipeline."""

    def __init__(
        self,
        transformers: list[BaseTransformer] | None = None,
    ) -> None:
        self.transformers = transformers or []

    def add(
        self,
        transformer: BaseTransformer,
    ) -> "PreprocessingPipeline":
        """Add a transformer to the pipeline."""
        self.transformers.append(transformer)
        return self

    def fit(self, df: pd.DataFrame) -> "PreprocessingPipeline":
        """Fit every transformer sequentially."""
        current_df = df.copy()

        for transformer in self.transformers:
            transformer.fit(current_df)
            current_df = transformer.transform(current_df)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply every fitted transformer sequentially."""
        current_df = df.copy()

        for transformer in self.transformers:
            current_df = transformer.transform(current_df)

        return current_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the complete pipeline."""
        self.fit(df)

        return self.transform(df)

    def get_config(self) -> list[dict[str, Any]]:
        """Return configurations of all transformers."""
        return [transformer.get_config() for transformer in self.transformers]

    def __len__(self) -> int:
        """Return the number of transformers."""
        return len(self.transformers)


def build_pipeline(
    config: PreprocessingConfig | dict[str, Any],
) -> PreprocessingPipeline:
    """Build a preprocessing pipeline from configuration."""
    if isinstance(config, dict):
        config = PreprocessingConfig.model_validate(config)

    pipeline = PreprocessingPipeline()

    if config.imputation.enabled:
        pipeline.add(
            MissingValueImputer(
                numerical_strategy=config.imputation.numerical_strategy,
                categorical_strategy=config.imputation.categorical_strategy,
            )
        )

    if config.encoding.enabled:
        pipeline.add(
            CategoricalEncoder(
                columns=config.encoding.columns,
                drop_first=config.encoding.drop_first,
            )
        )

    if config.scaling.enabled:
        pipeline.add(
            NumericalScaler(
                columns=config.scaling.columns,
                method=config.scaling.method,
            )
        )

    return pipeline
