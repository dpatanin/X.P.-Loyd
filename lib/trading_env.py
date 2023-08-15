from typing import TypedDict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from gymnasium import spaces
from tf_agents.specs import ArraySpec, BoundedArraySpec, BoundedTensorSpec
from tf_agents.trajectories import time_step as ts


class TradingStats(TypedDict):
    profit: float
    fees: float
    balance: float


class TradingEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int,
        features: list[str],
        trade_limit=10,
        balance=10000.00,
        tick_ratio=12.5 / 0.25,
        fees_per_contract=0.00,
    ):
        super(TradingEnvironment, self).__init__()

        self.df = df
        self.window_size = window_size
        self.features = features
        self.trade_limit = trade_limit
        self.prices = df[["high", "low", "open", "close"]]
        self.tick_ratio = tick_ratio
        self.fees_per_contract = fees_per_contract

        def act_pos_space():
            return spaces.Box(low=-trade_limit, high=trade_limit, shape=(1,))

        def simple_box(length):
            return spaces.Box(
                low=-np.inf, high=np.inf, shape=(length,), dtype=np.float32
            )

        self.action_space = act_pos_space()
        self.observation_space = spaces.Dict(
            {
                "position": act_pos_space(),
                "balance": simple_box(1),
            }
            | {f: simple_box(window_size) for f in features}
        )

        # episode
        self._init_balance = balance
        self._start_tick = self.window_size
        self._end_tick = len(self.df) - 1

        self.reset()

    def reset(self):
        self._done = False
        self._truncated = False
        self._balance = self._init_balance
        self._current_tick = self._start_tick
        self._entry_price = 0
        self._position = 0
        self._position_history = (self.window_size * [None]) + [self._position]
        self._first_rendering = True
        self.history = {}
        self._update_observation()

        return self.observation

    def step(self, action):
        self._current_tick += 1
        self._done = self._current_tick == self._end_tick
        trade_volume = round(action)

        price_diff = 0
        if trade_volume != self._position:
            current_close_price = self.df["close"].iloc[self._current_tick]
            if self._position == 1:
                price_diff = current_close_price - self.entry_price
            elif self._position == 2:
                price_diff = self.entry_price - current_close_price
            self.entry_price = current_close_price

        reward = self._calculate_reward(price_diff)
        profit, fees = self._update_balance(price_diff, trade_volume)
        self._truncated = self._balance <= 0

        self._position = action
        self._position_history.append(self._position)

        self._update_observation()
        info: TradingStats = {"profit": profit, "fees": fees, "balance": self._balance}
        self._update_history(info)

        return self.observation, reward, self._done, self._truncated, info

    def _update_observation(self):
        next_window = self.df[
            (self._current_tick - self.window_size + 1) : self._current_tick + 1
        ]

        # Ensure order of inputs/outputs
        self.observation = {
            "position": [self._position],
            "balance": [self._balance],
        } | {f: next_window[f].values for f in self.features}

        # np.concatenate(
        #     [
        #         [self._position],
        #         [self._balance],
        #         *[next_window[feature].values for feature in self.features],
        #     ]
        # )

    def _update_history(self, info: TradingStats):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _calculate_reward(self, diff_price: float):
        # TODO: Reward function?
        return diff_price

    def _update_balance(self, price_diff: float, trade_volume: int):
        fees = trade_volume * self.fees_per_contract
        profit = trade_volume * price_diff * self.tick_ratio
        self._balance += profit - fees
        return (profit, fees)

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

    def observation_spec(self):
        return {
            k: ArraySpec(v.shape, v.dtype)
            for k, v in self.observation_space.spaces.items()
        }

    def action_spec(self):
        if isinstance(self.action_space, gym.spaces.Box):
            return BoundedArraySpec(
                shape=self.action_space.shape,
                dtype=self.action_space.dtype,
                minimum=self.action_space.low,
                maximum=self.action_space.high,
            )
        elif isinstance(self.action_space, gym.spaces.Discrete):
            return BoundedTensorSpec(
                shape=(),
                dtype=np.int32,
                minimum=self.action_space.n - 1,
                maximum=self.action_space.n - 1,
            )

    def time_step_spec(self):
        return ts.time_step_spec(self.observation_spec())
