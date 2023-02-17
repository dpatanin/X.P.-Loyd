import pandas as pd

data_header = ["Open", "High", "Low", "Close", "Volume"]

def assert_columns(data: pd.DataFrame, required_columns=data_header):
    assert set(required_columns).issubset(
        data.columns
    ), f"DataFrame {data} does not contain the required columns: {required_columns}"
