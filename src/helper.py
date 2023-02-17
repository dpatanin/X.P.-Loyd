import pandas as pd

default_header = ["Open", "High", "Low", "Close", "Volume"]

def assert_columns(data: pd.DataFrame, required_columns: list[str]):
    assert set(required_columns).issubset(
        data.columns
    ), f"DataFrame {data} does not contain the required columns: {required_columns}"
