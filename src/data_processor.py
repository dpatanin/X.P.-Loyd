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
        dir: str,
        headers: list[str],
        sequence_length: int,
        batch_size: int,
    ):
        assert os.path.exists(dir), f"{dir} does not exist."

        self.dir = dir
        self.column_headers = headers
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.batched_dir = self.__batch_dir()

    def load_file(self, file_path: str) -> pd.DataFrame:
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
            for i in range(0, len(data), self.sequence_length)
            if len(data.iloc[i : i + self.sequence_length]) == self.sequence_length
        ]

    def assert_columns(self, data: pd.DataFrame):
        if missing_columns := set(self.column_headers) - set(data.columns):
            raise ValueError(
                f"DataFrame is missing required columns: {missing_columns}"
            )

    def __batch_dir(self):
        ls = os.listdir(self.dir)
        del ls[: (len(ls) % self.batch_size)]

        return [ls[i : i + self.batch_size] for i in range(0, len(ls), self.batch_size)]
