from typing import Any

import pandas as pd

from app.preprocessing.pipeline_builder import PreprocessingPipeline


class PipelineExecutor:
    """Execute preprocessing pipelines."""

    def __init__(self, pipeline: PreprocessingPipeline) -> None:
        self.pipeline = pipeline

    def execute(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Execute the pipeline and return the transformed dataset."""
        original_shape = df.shape

        transformed_df = self.pipeline.fit_transform(df)

        execution_summary = {
            "original_rows": original_shape[0],
            "original_columns": original_shape[1],
            "final_rows": transformed_df.shape[0],
            "final_columns": transformed_df.shape[1],
            "pipeline": self.pipeline.get_config(),
        }

        return transformed_df, execution_summary
