from os.path import exists
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import wget


class DataProcessor:
    """
    Loads data from file & processes it.
    Splits into train (70%), validation (20%) & test (10%) datasets.
    """

    def __init__(self, src: str) -> None:
        if self.is_local(src):
            df = pd.read_csv(src)
        else:
            df = pd.read_csv(wget.download(src))

        # close of t1 == open of t2 -> redundant inputs
        df.drop("open")
        # Transform datetime to sin & cos
        date_time = pd.to_datetime(df.pop("dateTime"), format="%Y-%m-%d %H:%M:%S")
        timestamp_s = date_time.map(pd.Timestamp.timestamp)

        df["Day sin"] = np.sin(timestamp_s * (2 * np.pi / 60))
        df["Day cos"] = np.cos(timestamp_s * (2 * np.pi / 60))

        # Split data
        n = len(df)
        train_df = df[: int(n * 0.7)]
        val_df = df[int(n * 0.7) : int(n * 0.9)]
        test_df = df[int(n * 0.9) :]

        self.num_features = df.shape[1]

        # Standardization; EMA used during training for further normalization
        train_mean = train_df.mean()
        train_std = train_df.std()

        self.train_df = (train_df - train_mean) / train_std
        self.val_df = (val_df - train_mean) / train_std
        self.test_df = (test_df - train_mean) / train_std

    def is_local(self, url):
        url_parsed = urlparse(url)
        return exists(url_parsed.path) if url_parsed.scheme in ("file", "") else False
