from typing import Any

import pandas as pd

from app.preprocessing.pipeline_builder import PreprocessingPipeline


class PipelineExecutor:
    """Execute preprocessing pipelines."""

    def __init__(
        self,
        pipeline: PreprocessingPipeline,
    ) -> None:
        self.pipeline = pipeline

    def execute(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Execute the pipeline and return transformed data and metadata."""
        original_shape = df.shape
        original_missing = int(df.isna().sum().sum())
        original_columns = df.columns.tolist()

        transformed_df = self.pipeline.fit_transform(df)

        final_shape = transformed_df.shape
        final_missing = int(transformed_df.isna().sum().sum())

        execution_summary = {
            "original_rows": original_shape[0],
            "original_columns": original_shape[1],
            "final_rows": final_shape[0],
            "final_columns": final_shape[1],
            "original_missing_values": original_missing,
            "final_missing_values": final_missing,
            "columns_added": [
                column
                for column in transformed_df.columns
                if column not in original_columns
            ],
            "columns_removed": [
                column
                for column in original_columns
                if column not in transformed_df.columns
            ],
            "pipeline": self.pipeline.get_config(),
        }

        return transformed_df, execution_summary
