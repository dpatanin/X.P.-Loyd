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

    def __init__(self, src: str, ema_period: int) -> None:
        if self.is_local(src):
            df = pd.read_csv(src)
        else:
            file_path = wget.download(src)
            df = pd.read_csv(file_path)
            os.remove(file_path)

        print("Processing data...")
        self.num_features = df.shape[1]

        # Transform dateTime into periodic frequencies
        date_time = pd.to_datetime(df.pop("dateTime"), format="%Y-%m-%d %H:%M:%S")
        timestamp_s = date_time.map(pd.Timestamp.timestamp)
        day = 24 * 60 * 60

        df["day_sin"] = np.sin(timestamp_s * (2 * np.pi / day))
        df["day_cos"] = np.cos(timestamp_s * (2 * np.pi / day))

        # Preprocess price data
        for col in ["high", "low"]:
            df[col] = (df[col] - df["close"]).abs()

        for col in ["open", "close"]:
            series = df.pop(col)
            df[f"{col}_pct"] = series.pct_change()
            df[f"{col}_ema"] = series.ewm(span=ema_period, adjust=False).mean()

        df.fillna(0, inplace=True)

        # Split data
        n = len(df)
        self.train_df = df[: int(n * 0.7)]
        self.val_df = df[int(n * 0.7) : int(n * 0.9)]
        self.test_df = df[int(n * 0.9) :]

    def is_local(self, url):
        url_parsed = urlparse(url)
        return exists(url_parsed.path) if url_parsed.scheme in ("file", "") else False