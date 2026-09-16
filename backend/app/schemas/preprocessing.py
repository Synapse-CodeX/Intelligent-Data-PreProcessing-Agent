from typing import Any

from pydantic import BaseModel, Field


class ImputationConfig(BaseModel):
    """Configuration for missing-value imputation."""

    enabled: bool = False
    numerical_strategy: str = "median"
    categorical_strategy: str = "most_frequent"


class EncodingConfig(BaseModel):
    """Configuration for categorical encoding."""

    enabled: bool = False
    columns: list[str] | None = None
    drop_first: bool = True


class ScalingConfig(BaseModel):
    """Configuration for numerical scaling."""

    enabled: bool = False
    columns: list[str] | None = None
    method: str = "standard"


class PreprocessingConfig(BaseModel):
    """Complete preprocessing pipeline configuration."""

    imputation: ImputationConfig = Field(default_factory=ImputationConfig)

    encoding: EncodingConfig = Field(default_factory=EncodingConfig)

    scaling: ScalingConfig = Field(default_factory=ScalingConfig)

    metadata: dict[str, Any] = Field(default_factory=dict)
