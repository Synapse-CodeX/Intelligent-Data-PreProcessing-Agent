import pandas as pd


def infer_data_types(df: pd.DataFrame) -> dict[str, str]:
    """Infer the pandas data type of each dataset column."""
    return {str(column): str(df[column].dtype) for column in df.columns}


def infer_semantic_types(
    df: pd.DataFrame,
    numeric_threshold: float = 0.8,
    datetime_threshold: float = 0.8,
) -> dict[str, str]:
    """
    Infer semantic data types from column values.

    Semantic types describe what the data represents rather than
    only how pandas stores the values.
    """
    semantic_types: dict[str, str] = {}

    for column in df.columns:
        series = df[column].dropna()

        if series.empty:
            semantic_types[str(column)] = "unknown"
            continue

        dtype = series.dtype

        if pd.api.types.is_bool_dtype(dtype):
            semantic_types[str(column)] = "boolean"
            continue

        if pd.api.types.is_numeric_dtype(dtype):
            semantic_types[str(column)] = "numeric"
            continue

        if pd.api.types.is_datetime64_any_dtype(dtype):
            semantic_types[str(column)] = "datetime"
            continue

        string_series = series.astype(str).str.strip()

        numeric_values = pd.to_numeric(
            string_series,
            errors="coerce",
        )

        numeric_ratio = float(numeric_values.notna().mean())

        if numeric_ratio >= numeric_threshold:
            semantic_types[str(column)] = "numeric"
            continue

        datetime_values = pd.to_datetime(
            string_series,
            errors="coerce",
            format="mixed",
        )

        datetime_ratio = float(datetime_values.notna().mean())

        if datetime_ratio >= datetime_threshold:
            semantic_types[str(column)] = "datetime"
            continue

        semantic_types[str(column)] = "categorical"

    return semantic_types
