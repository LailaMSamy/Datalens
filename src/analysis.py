import pandas as pd


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    stats = numeric.describe().T.reset_index()
    stats = stats.rename(columns={"index": "feature"})
    return stats


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return pd.DataFrame()
    return numeric.corr()


def grouped_summary(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    out = df.groupby(group_col, dropna=False)[value_col].agg(["mean", "sum", "count"]).reset_index()
    return out.round(2)


def time_trend(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    temp = df[[date_col, value_col]].dropna().copy()
    temp = temp.sort_values(date_col)
    return temp.groupby(date_col)[value_col].mean().reset_index()


def top_correlations(corr_df: pd.DataFrame) -> pd.DataFrame:
    if corr_df.empty:
        return pd.DataFrame()

    rows = []
    cols = corr_df.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rows.append({
                "feature_1": cols[i],
                "feature_2": cols[j],
                "correlation": corr_df.iloc[i, j]
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["abs_corr"] = out["correlation"].abs()
    out = out.sort_values("abs_corr", ascending=False).drop(columns=["abs_corr"])
    return out.reset_index(drop=True)