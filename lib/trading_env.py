from typing import TypedDict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces

from lib.ensemble import Ensemble


class TradingStats(TypedDict):
    profit: float
    fees: float


class TradingEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int,
        forecast_cb: Ensemble.forecast,
        forecast_length: int,
        balance=1000.00,
        tick_ratio=12.5 / 0.25,
        fees_per_contract=0.00,
        trade_volume=1,
    ):
        super(TradingEnvironment, self).__init__()

        self.df = df
        self.forecast_cb = forecast_cb
        self.window_size = window_size
        self.prices = df[["high", "low", "open", "close"]]

        self.tick_ratio = tick_ratio
        self.fees_per_contract = fees_per_contract
        self.trade_volume = trade_volume  # TODO: Dynamic volumes?

        # Actions & Position: 0 = No position; 1 = Long; 2 = Short
        self.action_space = spaces.Discrete(3)

        def forecast_box():
            return spaces.Box(
                low=-np.inf, high=np.inf, shape=(forecast_length,), dtype=np.float32
            )

        self.observation_space = spaces.Dict(
            {
                "lstm_forecast": forecast_box(),
                "ar_forecast": forecast_box(),
                "gru_forecast": forecast_box(),
                "position": spaces.Discrete(3),
                "balance": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
            }
        )
        self.state_dim = sum(
            np.prod(obs.shape) for obs in self.observation_space.values()
        )

        # episode
        self._init_balance = balance
        self._start_tick = self.window_size
        self._end_tick = len(self.df) - 1
        self._reset_tracking()

    def _reset_tracking(self):
        self._done = False
        self._truncated = False
        self._balance = self._init_balance
        self._current_tick = self._start_tick
        self._entry_price = 0
        self._position = 0
        self._position_history = (self.window_size * [None]) + [self._position]
        self._first_rendering = True
        self.history = {}

    def reset(self, *, seed=None, options=None):
        self._reset_tracking()
        return self._get_observation(), {}

    def step(self, action):
        self._current_tick += 1
        self._done = self._current_tick == self._end_tick

        price_diff = 0
        if action != self._position:
            current_close_price = self.df["close"].iloc[self._current_tick]
            if self._position == 1:
                price_diff = current_close_price - self.entry_price
            elif self._position == 2:
                price_diff = self.entry_price - current_close_price
            self.entry_price = current_close_price

        reward = self._calculate_reward(price_diff)
        profit, fees = self._update_balance(price_diff)
        self._truncated = self._balance <= 0

        self._position = action
        self._position_history.append(self._position)
        observation = self._get_observation()
        info: TradingStats = {
            "profit": profit,
            "fees": fees,
        }
        self._update_history(info)

        return observation, reward, self._done, self._truncated, info

    def _get_observation(self):
        next_window = self.df[
            (self._current_tick - self.window_size + 1) : self._current_tick + 1
        ]
        forecast = self.forecast_cb(next_window)

        return {
            "lstm_forecast": forecast["lstm"],
            "ar_forecast": forecast["ar"],
            "gru_forecast": forecast["gru"],
            "position": self._position,
            "balance": [self._balance],
        }

    def _update_history(self, info: TradingStats):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render(self, mode="human"):
        def _plot_position(position, tick):
            if position == 0:
                color = "blue"
            elif position == 1:
                color = "green"
            elif position == 2:
                color = "red"
            else:
                color = None
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)
        plt.pause(0.01)

    def render_all(self, mode="human"):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == 2:
                short_ticks.append(tick)
            elif self._position_history[i] == 1:
                long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], "ro")
        plt.plot(long_ticks, self.prices[long_ticks], "go")

    def close(self):
        plt.close()

    def save_rendering(self, filepath: str):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _calculate_reward(self, diff_price: float):
        # TODO: Reward function?
        return diff_price

    def _update_balance(self, price_diff: float):
        fees = self.trade_volume * self.fees_per_contract
        profit = self.trade_volume * price_diff * self.tick_ratio
        self._balance += profit - fees
        return (profit, fees)
