import pandas as pd
import os


def load_data(file_path: str) -> pd.DataFrame:
    assert os.path.exists(file_path), f"{file_path} does not exist."
    data = pd.read_csv(file_path)

    # Check for the presence of specific columns
    # TODO: DateTime mandatory if data processed by using datetime
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    assert set(required_columns).issubset(
        data.columns
    ), f"File {file_path} does not contain the required columns: {required_columns}"

    return data
