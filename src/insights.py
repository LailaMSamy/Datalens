import pandas as pd
from src.analysis import top_correlations


def explain_dataset(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str], datetime_cols: list[str]) -> list[str]:
    notes = []
    notes.append(f"This dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    notes.append(f"It includes {len(numeric_cols)} numeric features, {len(categorical_cols)} categorical features, and {len(datetime_cols)} datetime features.")

    if datetime_cols:
        notes.append("Because a datetime field is available, the dataset supports trend analysis over time.")
    if categorical_cols:
        notes.append("Categorical fields can be used to compare performance across groups such as product types, regions, or segments.")
    if numeric_cols:
        notes.append("Numeric fields support statistical analysis, correlation analysis, clustering, and anomaly detection.")
    return notes


def generate_insights(df: pd.DataFrame, missing_df: pd.DataFrame, corr_df: pd.DataFrame) -> list[str]:
    insights = []

    high_missing = missing_df[missing_df["missing_percent"] > 20]
    if not high_missing.empty:
        insights.append(
            f"{len(high_missing)} columns contain more than 20% missing values, which may weaken downstream analysis."
        )

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        means = numeric.mean().sort_values(ascending=False)
        insights.append(
            f"The feature with the highest average value is '{means.index[0]}', with a mean of {means.iloc[0]:.2f}."
        )

    corr_pairs = top_correlations(corr_df)
    if not corr_pairs.empty:
        best = corr_pairs.iloc[0]
        insights.append(
            f"The strongest linear relationship appears between '{best['feature_1']}' and '{best['feature_2']}' with a correlation of {best['correlation']:.2f}."
        )

    if numeric.shape[1] >= 1:
        max_std_col = numeric.std().sort_values(ascending=False).index[0]
        insights.append(
            f"'{max_std_col}' shows the highest variability, suggesting stronger dispersion across observations."
        )

    if not insights:
        insights.append("The dataset is limited for advanced analysis, so richer features or additional records may be needed.")

    return insights