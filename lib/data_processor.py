import pandas as pd
import os


class DataProcessor:
    """
    Loads, processes and serves the data.\n
    It divides the sessions inside `dir` into batches of `batch_size` for parallel processing. (Leftover files are disregarded)
    When loading a batch, the time sequenced data is split into `sequence_length` long sequences. (Leftover sequences are disregarded)

    |`headers`: Specifies column header of csv files. Unspecified columns are dropped.
    """

    def __init__(
        self,
        headers: list[str],
        sequence_length: int,
        batch_size: int,
        step_size: int = None,
        dir: str = None,
    ):

        self.dir = dir
        self.column_headers = headers
        self.sequence_length = sequence_length
        self.step_size = step_size or sequence_length
        self.batch_size = batch_size
        self.batched_dir: list[list[str]] = self.batch_dir() if dir else [[]]

    def load_file(self, file_path: str) -> pd.DataFrame:
        # sourcery skip: pandas-avoid-inplace
        data = pd.read_csv(file_path)

        self.assert_columns(data)
        data.drop(set(data.columns) - set(self.column_headers), axis=1, inplace=True)

        return data

    def load_batch(self, batch_index: int):
        seq_files = [
            self.sequence(self.load_file(f"{self.dir}/{path}"))
            for path in self.batched_dir[batch_index]
        ]

        return [list(s) for s in zip(*seq_files)]

    def sequence(self, data: pd.DataFrame) -> list[pd.DataFrame]:
        self.assert_columns(data)
        return [
            data.iloc[i : i + self.sequence_length]
            for i in range(0, len(data), self.step_size)
            if len(data.iloc[i : i + self.sequence_length]) == self.sequence_length
        ]

    def assert_columns(self, data: pd.DataFrame):
        if missing_columns := set(self.column_headers) - set(data.columns):
            raise ValueError(
                f"DataFrame is missing required columns: {missing_columns}"
            )

    def batch_dir(self):
        assert os.path.exists(self.dir), f"{self.dir} does not exist."
        ls = os.listdir(self.dir)
        num_batches = len(ls) // self.batch_size
        remaining_paths = len(ls) % self.batch_size

        batches = [ls[i:i + self.batch_size] for i in range(0, num_batches * self.batch_size, self.batch_size)]
        if remaining_paths:
            batches.append(ls[-remaining_paths:])

        return batches
