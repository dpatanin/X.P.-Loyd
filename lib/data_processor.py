import os
from os.path import exists
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import wget
from tqdm import tqdm


class DataProcessor:
    """
    Loads data from file & processes it.
    Splits into train (70%), validation (20%) & test (10%) datasets.
    """

    def __init__(self, src: str, period_fast=3, period_slow=5) -> None:
        download = None if self._is_local(src) else wget.download(src)

        self._pb = tqdm(range(7), desc="Load data")
        df = pd.read_csv(download or src)
        if download:
            os.remove(download)

        self.num_features = df.shape[1]

        self._update_pb("Transform dateTime")
        # Transform dateTime into periodic frequencies
        date_time = pd.to_datetime(df.pop("dateTime"), format="%Y-%m-%d %H:%M:%S")
        self._update_pb()

        timestamp_s = date_time.astype(np.int64) // 10**9
        self._update_pb()

        day = 24 * 60 * 60
        df["day_sin"] = np.sin(timestamp_s * (2 * np.pi / day))
        df["day_cos"] = np.cos(timestamp_s * (2 * np.pi / day))
        df.set_index(date_time, inplace=True)

        self._update_pb("Process prices")
        # Preprocess price data
        for col in ["high", "low"]:
            df[f"{col}_diff"] = (df[col] - df["close"]).abs()

        for col in ["open", "close"]:
            df[f"{col}_pct"] = df[col].pct_change()
            df[f"{col}_diff"] = df[col].diff()

        self._update_pb("Calculate SMA Crossover")
        df["SMA_diff"] = self.sma(df["close"], period_fast) / self.sma(
            df["close"], period_slow
        )
        df["SMA_position"] = np.where(df["SMA_diff"] > 1, 1, 2)

        df.fillna(0, inplace=True)

        self._update_pb("Split data")
        # Split data
        n = len(df)
        self.train_df = df[: int(n * 0.7)]
        self.val_df = df[int(n * 0.7) : int(n * 0.9)]
        self.test_df = df[int(n * 0.9) :]
        self._update_pb("Data processed!")
        self._pb.close()

    def rma(self, s: pd.Series, period: int) -> pd.Series:
        return s.ewm(alpha=1 / period).mean()

    def atr(self, df: pd.DataFrame, length: int = 14) -> pd.Series:
        # Ref: https://stackoverflow.com/a/74282809/
        high, low, prev_close = df["high"], df["low"], df["close"].shift()
        tr_all = [high - low, high - prev_close, low - prev_close]
        tr_all = [tr.abs() for tr in tr_all]
        tr = pd.concat(tr_all, axis=1).max(axis=1)
        return self.rma(tr, length)

    def sma(self, s: pd.Series, period: int) -> pd.Series:
        return s.rolling(window=period).mean()

    def _is_local(self, url):
        url_parsed = urlparse(url)
        return exists(url_parsed.path) if url_parsed.scheme in ("file", "") else False

    def _update_pb(self, desc: str = None):
        self._pb.update()
        if desc:
            self._pb.set_description(desc)
