import pandas as pd
import os
from src.helper import assert_columns, default_header


class DataProcessor:
    def __init__(
        self,
        file_path: str,
        window_size: int,
        slide_length: int = None,
        column_header: list[str] = default_header,
    ):
        self.column_header = column_header
        self.load_data(file_path)
        self.window_data(
            self.data, window_size, slide_length=slide_length or window_size
        )

    def load_data(self, file_path: str) -> pd.DataFrame:
        assert os.path.exists(file_path), f"{file_path} does not exist."
        data = pd.read_csv(file_path)
        assert_columns(data, self.column_header)
        
        data.drop("DateTime", axis=1, inplace=True)

        self.data = data
        return data

    def window_data(
        self, data: pd.DataFrame, window_size: int, slide_length: int
    ) -> list[pd.DataFrame]:
        assert_columns(data, self.column_header)

        windowed_data: list[pd.DataFrame] = []

        i = 0
        while i < len(data):
            upper_boundary = (
                i + window_size if i + window_size < len(data) else len(data) - 1
            )

            windowed_data.append(data.iloc[i : upper_boundary, :])
            i += slide_length

        self.windowed_data = windowed_data
        return windowed_data
