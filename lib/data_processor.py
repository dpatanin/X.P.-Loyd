import os
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
            file_path = wget.download(src)
            df = pd.read_csv(file_path)
            os.remove(file_path)

        # Transform periodic frequencies
        date_time = pd.to_datetime(df.pop("dateTime"), format="%Y-%m-%d %H:%M:%S")
        timestamp_s = date_time.map(pd.Timestamp.timestamp)

        df["day_sin"] = np.sin(timestamp_s * (2 * np.pi / 60))
        df["day_cos"] = np.cos(timestamp_s * (2 * np.pi / 60))

        volumes = df.pop("volume")
        df["volume_sin"] = np.sin(volumes * (2 * np.pi / 60))
        df["volume_cos"] = np.cos(volumes * (2 * np.pi / 60))

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
