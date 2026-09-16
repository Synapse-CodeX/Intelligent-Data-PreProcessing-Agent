import pandas as pd


def infer_data_types(df: pd.DataFrame) -> dict[str, str]:
    """Infer the pandas data type of each dataset column."""
    return {column: str(df[column].dtype) for column in df.columns}
