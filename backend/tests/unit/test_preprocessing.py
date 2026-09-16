import numpy as np
import pandas as pd
from app.preprocessing.encoding import CategoricalEncoder
from app.preprocessing.feature_engineering import (
    NumericInteractionTransformer,
)
from app.preprocessing.feature_selection import ConstantFeatureRemover
from app.preprocessing.imputation import MissingValueImputer
from app.preprocessing.outlier_handling import IQRWinsorizer
from app.preprocessing.pipeline_builder import (
    PreprocessingPipeline,
    build_pipeline,
)
from app.preprocessing.pipeline_executor import PipelineExecutor
from app.preprocessing.scaling import NumericalScaler
from app.preprocessing.transformations import LogTransformer
from app.schemas.preprocessing import PreprocessingConfig


def test_missing_value_imputer():
    df = pd.DataFrame(
        {
            "age": [20, 30, None, 40],
            "city": ["Kolkata", "Delhi", None, "Mumbai"],
        }
    )

    transformer = MissingValueImputer()
    result = transformer.fit_transform(df)

    assert result["age"].isna().sum() == 0
    assert result["city"].isna().sum() == 0
    assert result.loc[2, "age"] == 30


def test_categorical_encoder():
    df = pd.DataFrame(
        {
            "city": ["Kolkata", "Delhi", "Mumbai"],
        }
    )

    transformer = CategoricalEncoder()
    result = transformer.fit_transform(df)

    assert result.shape[1] == 2
    assert all(column.startswith("city_") for column in result.columns)


def test_numerical_scaler():
    df = pd.DataFrame(
        {
            "value": [10.0, 20.0, 30.0],
        }
    )

    transformer = NumericalScaler(method="standard")
    result = transformer.fit_transform(df)

    assert np.isclose(result["value"].mean(), 0.0)
    assert np.isclose(result["value"].std(ddof=0), 1.0)


def test_minmax_scaler():
    df = pd.DataFrame(
        {
            "value": [10.0, 20.0, 30.0],
        }
    )

    transformer = NumericalScaler(method="minmax")
    result = transformer.fit_transform(df)

    assert result["value"].min() == 0.0
    assert result["value"].max() == 1.0


def test_log_transformer():
    df = pd.DataFrame(
        {
            "value": [0.0, 1.0, 10.0],
        }
    )

    transformer = LogTransformer()
    result = transformer.fit_transform(df)

    assert np.isclose(
        result.loc[0, "value"],
        0.0,
    )
    assert np.isclose(
        result.loc[1, "value"],
        np.log1p(1.0),
    )


def test_log_transformer_rejects_negative_values():
    df = pd.DataFrame(
        {
            "value": [-1.0, 0.0, 10.0],
        }
    )

    transformer = LogTransformer()

    try:
        transformer.fit_transform(df)
        raise AssertionError("Expected ValueError for negative values.")
    except ValueError:
        pass


def test_iqr_winsorizer():
    df = pd.DataFrame(
        {
            "value": [10.0, 11.0, 12.0, 13.0, 100.0],
        }
    )

    transformer = IQRWinsorizer()
    result = transformer.fit_transform(df)

    assert result["value"].max() < 100.0


def test_constant_feature_remover():
    df = pd.DataFrame(
        {
            "constant": [1, 1, 1],
            "value": [1, 2, 3],
        }
    )

    transformer = ConstantFeatureRemover()
    result = transformer.fit_transform(df)

    assert "constant" not in result.columns
    assert "value" in result.columns


def test_preprocessing_pipeline():
    df = pd.DataFrame(
        {
            "age": [20.0, 30.0, None],
            "city": ["Kolkata", "Delhi", "Kolkata"],
        }
    )

    pipeline = (
        PreprocessingPipeline()
        .add(MissingValueImputer())
        .add(CategoricalEncoder())
        .add(NumericalScaler())
    )

    result = pipeline.fit_transform(df)

    assert result.isna().sum().sum() == 0
    assert "city_Kolkata" in result.columns
    assert "city_Delhi" not in result.columns


def test_pipeline_executor():
    df = pd.DataFrame(
        {
            "age": [20.0, 30.0, None],
            "city": ["Kolkata", "Delhi", "Kolkata"],
        }
    )

    pipeline = (
        PreprocessingPipeline().add(MissingValueImputer()).add(CategoricalEncoder())
    )

    executor = PipelineExecutor(pipeline)

    result, summary = executor.execute(df)

    assert result.shape[0] == 3
    assert summary["original_rows"] == 3
    assert summary["final_rows"] == 3
    assert len(summary["pipeline"]) == 2
    assert summary["original_missing_values"] == 1
    assert summary["final_missing_values"] == 0


def test_numeric_interaction_transformer():
    df = pd.DataFrame(
        {
            "income": [10.0, 20.0],
            "purchase": [2.0, 3.0],
        }
    )

    transformer = NumericInteractionTransformer(columns=["income", "purchase"])

    result = transformer.fit_transform(df)

    assert "income_x_purchase" in result.columns
    assert result.loc[0, "income_x_purchase"] == 20.0
    assert result.loc[1, "income_x_purchase"] == 60.0


def test_preprocessing_config():
    config = PreprocessingConfig(
        imputation={
            "enabled": True,
            "numerical_strategy": "median",
        },
        encoding={
            "enabled": True,
            "drop_first": True,
        },
        scaling={
            "enabled": True,
            "method": "standard",
        },
    )

    assert config.imputation.enabled is True
    assert config.encoding.enabled is True
    assert config.scaling.method == "standard"


def test_build_pipeline_from_config():
    config = PreprocessingConfig(
        imputation={
            "enabled": True,
        },
        encoding={
            "enabled": True,
        },
        scaling={
            "enabled": True,
        },
    )

    pipeline = build_pipeline(config)

    assert len(pipeline) == 3
    assert pipeline.get_config()[0]["name"] == ("missing_value_imputation")
    assert pipeline.get_config()[1]["name"] == ("categorical_encoding")
    assert pipeline.get_config()[2]["name"] == ("numerical_scaling")


def test_build_pipeline_from_dictionary():
    config = {
        "imputation": {
            "enabled": True,
        },
        "encoding": {
            "enabled": True,
        },
    }

    pipeline = build_pipeline(config)

    assert len(pipeline) == 2


def test_configured_pipeline_execution():
    df = pd.DataFrame(
        {
            "age": [20.0, 30.0, None],
            "city": ["Kolkata", "Delhi", "Kolkata"],
        }
    )

    config = PreprocessingConfig(
        imputation={
            "enabled": True,
        },
        encoding={
            "enabled": True,
        },
    )

    pipeline = build_pipeline(config)

    executor = PipelineExecutor(pipeline)

    result, summary = executor.execute(df)

    assert result.isna().sum().sum() == 0
    assert result.shape[0] == 3
    assert len(summary["pipeline"]) == 2
