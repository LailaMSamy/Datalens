import pandas as pd


def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


def load_demo_data(path: str = "data/sample_data.csv") -> pd.DataFrame:
    return pd.read_csv(path)