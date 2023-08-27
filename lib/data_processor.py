import os
from os.path import exists
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import wget
from SMACrossOver import SMACrossOver
from tqdm import tqdm


class DataProcessor:
    """
    Loads data from file & processes it.
    Splits into train (70%), validation (20%) & test (10%) datasets.
    """

    def __init__(self, src: str, ema_period: int) -> None:
        download = None if self._is_local(src) else wget.download(src)

        self._pb = tqdm(range(10), desc="Loading data")
        df = pd.read_csv(download or src)
        if download:
            os.remove(download)

        self.num_features = df.shape[1]

        self._update_pb("Transforming dateTime")
        # Transform dateTime into periodic frequencies
        date_time = pd.to_datetime(df.pop("dateTime"), format="%Y-%m-%d %H:%M:%S")
        self._update_pb()

        timestamp_s = date_time.astype(np.int64) // 10**9
        self._update_pb()

        day = 24 * 60 * 60
        df["day_sin"] = np.sin(timestamp_s * (2 * np.pi / day))
        df["day_cos"] = np.cos(timestamp_s * (2 * np.pi / day))
        df.set_index(date_time, inplace=True)

        self._update_pb("Processing prices")
        # Preprocess price data
        for col in ["high", "low"]:
            self._update_pb()
            df[f"{col}_diff"] = (df[col] - df["close"]).abs()

        for col in ["open", "close"]:
            self._update_pb()
            df[f"{col}_pct"] = df[col].pct_change()
            df[f"{col}_ema"] = df[col].ewm(span=ema_period, adjust=False).mean()

        df.fillna(0, inplace=True)

        strategy = SMACrossOver(df)
        strategy.analyze(periodFast=3, periodSlow=5)

        self._update_pb("Splitting data")
        # Split data
        n = len(df)
        self.train_df = df[: int(n * 0.7)]
        self.val_df = df[int(n * 0.7) : int(n * 0.9)]
        self.test_df = df[int(n * 0.9) :]
        self._update_pb("Data processed!")
        self._pb.close()

    def _is_local(self, url):
        url_parsed = urlparse(url)
        return exists(url_parsed.path) if url_parsed.scheme in ("file", "") else False

    def _update_pb(self, desc: str = None):
        self._pb.update()
        if desc:
            self._pb.set_description(desc)
