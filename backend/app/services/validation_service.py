from typing import Any

import pandas as pd

from app.validation.comparison import DatasetComparison
from app.validation.dataset_validator import DatasetValidator
from app.validation.leakage_validator import LeakageValidator
from app.validation.preprocessing_validator import (
    PreprocessingValidator,
)
from app.validation.schema_validator import SchemaValidator


class ValidationService:
    """Orchestrate dataset validation."""

    def __init__(self) -> None:
        self.dataset_validator = DatasetValidator()
        self.preprocessing_validator = PreprocessingValidator()
        self.schema_validator = SchemaValidator()
        self.leakage_validator = LeakageValidator()
        self.comparison = DatasetComparison()

    def validate(
        self,
        original_df: pd.DataFrame,
        processed_df: pd.DataFrame,
        target_column: str | None = None,
    ) -> dict[str, Any]:
        """Run all validation checks."""
        dataset_result = self.dataset_validator.validate(processed_df)

        preprocessing_result = self.preprocessing_validator.validate(
            original_df,
            processed_df,
        )

        schema_result = self.schema_validator.validate(
            original_df,
            processed_df,
        )

        leakage_result = self.leakage_validator.validate(
            processed_df,
            target_column,
        )

        comparison_result = self.comparison.compare(
            original_df,
            processed_df,
        )

        validators = [
            dataset_result,
            preprocessing_result,
            schema_result,
            leakage_result,
        ]

        overall_passed = all(result["passed"] for result in validators)

        return {
            "passed": overall_passed,
            "dataset": dataset_result,
            "preprocessing": preprocessing_result,
            "schema": schema_result,
            "leakage": leakage_result,
            "comparison": comparison_result,
        }
