import pandas as pd


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = clean_column_names(df)
    df = try_parse_dates(df)
    df = df.drop_duplicates()
    return df


def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df) * 100).round(2)
    out = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.values,
        "missing_percent": missing_percent.values
    }).sort_values("missing_percent", ascending=False)
    return out.reset_index(drop=True)


def dataset_quality_report(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "missing": int(df.isna().sum().sum()),
    }