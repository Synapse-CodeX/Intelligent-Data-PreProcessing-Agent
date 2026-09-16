import pandas as pd


def calculate_correlations(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Calculate Pearson correlations between numerical columns."""
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return {}

    correlation_matrix = numeric_df.corr(method="pearson")

    return {
        column: {
            other_column: float(correlation_matrix.loc[column, other_column])
            for other_column in correlation_matrix.columns
        }
        for column in correlation_matrix.index
    }
