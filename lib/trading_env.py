from enum import Enum

import gym
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces

from lib.ensemble import Ensemble


class TradingEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        ensemble: Ensemble,
        balance=1000,
        tick_ratio=12.5 / 0.25,
        fees_per_contract=0,
        trade_volume=1,
    ):
        super(TradingEnvironment, self).__init__()

        self.df = df
        self.ensemble = ensemble
        self.window_size = ensemble.max_window_size
        self.prices = df[["high", "low", "open", "close"]]

        self.tick_ratio = tick_ratio
        self.fees_per_contract = fees_per_contract
        self.trade_volume = trade_volume  # TODO: Dynamic volumes?

        # Actions & Position: 0 = No position; 1 = Long; 2 = Short
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict(
            {
                "lstm_forecast": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(ensemble.lstm_window,),
                    dtype=np.float32,
                ),
                "ar_forecast": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(ensemble.ar_window,),
                    dtype=np.float32,
                ),
                "gru_forecast": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(ensemble.gru_window),
                    dtype=np.float32,
                ),
                "position": spaces.Discrete(3),
                "balance": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
            }
        )

        # episode
        self._balance = self._init_balance = balance
        self._start_tick = self.window_size
        self._end_tick = len(self.df) - 1
        self._done = None
        self._current_tick = None
        self._entry_price = 0
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

    def reset(self):
        self._done = False
        self._balance = self._init_balance
        self._current_tick = self._start_tick
        self._entry_price = 0
        self._last_trade_tick = self._current_tick - 1
        self._position = 0
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.0
        self._total_profit = 1.0  # unit
        self._first_rendering = True
        self.history = {}
        return self._get_observation()

    def step(self, action):
        self._current_tick += 1

        price_diff = 0
        if action != self._position:
            current_close_price = self.df["close"].iloc[self._current_tick]
            if self._position == 1:
                price_diff = current_close_price - self.entry_price
            elif self._position == 2:
                price_diff = self.entry_price - current_close_price
            self.entry_price = current_close_price
            self._last_trade_tick = self._current_tick

        reward = self._calculate_reward(price_diff)
        self._total_reward += reward
        self._update_profit(price_diff)

        self._done = self._balance <= 0 or self._current_tick == self._end_tick

        self._position = action
        self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position.value,
        )
        self._update_history(info)

        return observation, reward, self._done, info

    def _get_observation(self):
        next_window = self.df[
            (self._current_tick - self.window_size + 1) : self._current_tick + 1
        ]
        forecast = self.ensemble.forecast(next_window)

        return {
            "lstm_forecast": forecast["lstm"],
            "ar_forecast": forecast["ar"],
            "gru_forecast": forecast["gru"],
            "position": self._position,
            "balance": self._balance,
        }

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render(self, mode="human"):
        def _plot_position(position, tick):
            match position:
                case 0:
                    color = "blue"
                case 1:
                    color = "green"
                case 2:
                    color = "red"
                case _:
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

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward
            + " ~ "
            + "Total Profit: %.6f" % self._total_profit
        )

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

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward
            + " ~ "
            + "Total Profit: %.6f" % self._total_profit
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _calculate_reward(self, diff_price):
        # TODO: Reward function?
        return diff_price

    def _update_profit(self, reward):
        profit = self.trade_volume * (reward * self.tick_ratio - self.fees_per_contract)
        self._balance += profit
        self._total_profit += profit

    def max_possible_profit(self):  # trade fees are ignored
        # TODO: separate profits & fees -> track both for eval
        raise NotImplementedError
