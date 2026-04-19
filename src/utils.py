import pandas as pd


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def get_datetime_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()