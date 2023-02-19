import pandas as pd
import os
import random


class Data:
    """
    Processor: The processing unit for the data. It processes the data and splits and sorts it into the respective data bundles.
    Data chunks shorter than the specified parameter will be omitted.
    Raw: Entire data in one dataframe.
    Sequenced: Raw data split into sequences.
    Windowed: Sequenced data grouped into lists. Those lists may overlap.
    """
    def __init__(self, file_path: str, processor: "DataProcessor"):
        self.processor = processor
        self.raw = self.processor.load(file_path)
        self.sequenced = self.processor.sequence(self.raw)
        self.windowed = self.processor.window(self.sequenced)


class DataProcessor:
    def __init__(
        self,
        sequence_length: int,
        window_size: int,
        slide_length: int = None,
        column_headers: list[str] = None,
        dropped_headers: list[str] = None,
    ):
        self.sequence_length = sequence_length
        self.window_size = window_size
        self.slide_length = slide_length or window_size
        self.column_headers = column_headers or []
        self.dropped_headers = dropped_headers or []

    def load(self, file_path: str) -> pd.DataFrame:
        assert os.path.exists(file_path), f"{file_path} does not exist."
        data = pd.read_csv(file_path)
        self.assert_columns(data)

        data.drop(self.dropped_headers, axis=1, inplace=True)

        return data

    def sequence(self, data: pd.DataFrame) -> list[pd.DataFrame]:
        self.assert_columns(data)
        return [
            data.iloc[i : i + self.sequence_length]
            for i in range(0, len(data), self.sequence_length)
            if len(data.iloc[i : i + self.sequence_length]) == self.sequence_length
        ]

    def window(self, data: list[pd.DataFrame]) -> list[list[pd.DataFrame]]:
        self.assert_columns(data[random.randrange(len(data))])
        return [
            data[i : i + self.window_size]
            for i in range(0, len(data) - self.window_size + 1, self.slide_length)
            if len(data[i : i + self.window_size]) == self.window_size
        ]

    def assert_columns(self, data: pd.DataFrame):
        if missing_columns := set(self.column_headers) - set(data.columns):
            raise ValueError(
                f"DataFrame is missing required columns: {missing_columns}"
            )
