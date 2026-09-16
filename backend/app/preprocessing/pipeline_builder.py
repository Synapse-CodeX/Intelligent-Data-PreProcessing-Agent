from typing import Any

from app.preprocessing.base_transformer import BaseTransformer


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

    def fit(self, df):
        """Fit every transformer sequentially."""
        current_df = df.copy()

        for transformer in self.transformers:
            transformer.fit(current_df)
            current_df = transformer.transform(current_df)

        return self

    def transform(self, df):
        """Apply every fitted transformer sequentially."""
        current_df = df.copy()

        for transformer in self.transformers:
            current_df = transformer.transform(current_df)

        return current_df

    def fit_transform(self, df):
        """Fit and transform the complete pipeline."""
        self.fit(df)
        return self.transform(df)

    def get_config(self) -> list[dict[str, Any]]:
        """Return configurations of all transformers."""
        return [transformer.get_config() for transformer in self.transformers]
