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

    def __init__(self, src: str, period=14) -> None:
        download = None if self._is_local(src) else wget.download(src)

        self._pb = tqdm(range(8), desc="Load data")
        df = pd.read_csv(download or src)
        if download:
            os.remove(download)

        self.num_features = df.shape[1]

        # Transform dateTime into periodic frequencies
        self._update_pb("Transform dateTime")
        date_time = pd.to_datetime(df.pop("dateTime"), format="%Y-%m-%d %H:%M:%S")
        self._update_pb()

        timestamp_s = date_time.astype(np.int64) // 10**9
        self._update_pb()

        day = 24 * 60 * 60
        df["day_sin"] = np.sin(timestamp_s * (2 * np.pi / day))
        df["day_cos"] = np.cos(timestamp_s * (2 * np.pi / day))
        df.set_index(date_time, inplace=True)

        self._update_pb("Calculate RSS")
        df["rss"] = self.rss(df["close"])

        self._update_pb("Calculate DMI")
        df["dmi"] = self.dmi(df["close"], df["high"], df["low"], period)

        self._update_pb("Calculate double stochastic")
        df["double_stochastic"] = self.double_stochastic(
            df["close"], df["high"], df["low"], period
        )

        df.fillna(0, inplace=True)

        self._update_pb("Split data")
        # Split data
        n = len(df)
        self.train_df = df[: int(n * 0.7)]
        self.val_df = df[int(n * 0.7) : int(n * 0.9)]
        self.test_df = df[int(n * 0.9) :]
        self._update_pb("Data processed!")
        self._pb.close()

    def sma(self, s: pd.Series, period: int) -> pd.Series:
        return s.rolling(window=period).mean()

    def ema(self, s: pd.Series, period: int) -> pd.Series:
        return s.ewm(span=period).mean()

    def rma(self, s: pd.Series, period: int) -> pd.Series:
        return s.ewm(alpha=1 / period).mean()

    def rsi(self, s: pd.Series, period: int) -> pd.Series:
        change = s.diff()
        gain = self.rma(change.mask(change < 0, 0.0), period)
        loss = self.rma(-change.mask(change > 0, -0.0), period)

        return 100 - (100 / (1 + (gain / loss)))

    def rss(self, s: pd.Series) -> pd.Series:
        spread = self.ema(s, 10) - self.ema(s, 40)
        rsi = self.rsi(spread, 5)

        return self.sma(rsi, 5)

    def dmi(
        self, close: pd.Series, high: pd.Series, low: pd.Series, period: int
    ) -> pd.Series:
        sma_tr = self.sma(
            np.maximum(
                high - low, np.maximum((high - close).abs(), (low - close).abs())
            ),
            period,
        )
        low_diff = low.diff()
        low_diff = low_diff.mask(low_diff > 0, 0)
        high_diff = -high.diff()
        high_diff = high_diff.mask(high_diff > 0, 0)

        sma_dm_minus = self.sma(low_diff.mask(low_diff > high_diff, 0), period)
        sma_dm_plus = self.sma(high_diff.mask(high_diff > low_diff, 0), period)

        di_minus = sma_dm_minus / sma_tr
        di_plus = sma_dm_plus / sma_tr

        return (di_plus - di_minus) / (di_plus + di_minus)

    def double_stochastic(
        self, close: pd.Series, high: pd.Series, low: pd.Series, period: int
    ) -> pd.Series:
        max_high = high.rolling(window=period).max()
        min_low = low.rolling(window=period).min()

        pct_k = ((close - min_low) / (max_high - min_low)) * 100
        pct_d = self.sma(pct_k, period)

        diff = pct_k - pct_d

        return ((diff - diff.min()) / (diff.max() - diff.min())) * 100

    def _is_local(self, url):
        url_parsed = urlparse(url)
        return exists(url_parsed.path) if url_parsed.scheme in ("file", "") else False

    def _update_pb(self, desc: str = None):
        self._pb.update()
        if desc:
            self._pb.set_description(desc)
