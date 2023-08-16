from typing import Any, Optional, Text

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from gymnasium import spaces
from tf_agents.environments import PyEnvironment, TFEnvironment
from tf_agents.specs import BoundedArraySpec, array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common


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

        self.action_space = spaces.Box(low=-trade_limit, high=trade_limit, shape=(1,))

        obs_shape = (2 + window_size * len(features),)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float64)

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
        assert (
            action.shape == self.action_spec().shape
        ), f"Unexpected action shape.\nReceived: {action.shape}.\nExpected: {self.action_spec().shape}"
        self._current_tick += 1
        self._done = self._current_tick == self._end_tick
        trade_volume = round(action[0])

        profit = fees = 0.00
        if trade_volume != self._position:
            current_close_price = self.df["close"].iloc[self._current_tick]
            price_diff = current_close_price - self._entry_price
            self._entry_price = current_close_price

            profit, fees = self._update_balance(price_diff, self._position)
            self._truncated = self._balance <= 0

        self._position = trade_volume
        self._position_history.append(self._position)

        self._update_observation()
        info = {"profit": profit, "fees": fees, "balance": self._balance}
        self._update_history(info)

        return self.observation, profit, self._done, self._truncated, info

    def _update_observation(self):
        next_window = self.df[
            (self._current_tick - self.window_size + 1) : self._current_tick + 1
        ]

        # Ensure order of inputs/outputs
        self.observation = np.concatenate(
            [
                [self._position],
                [self._balance],
                *[next_window[feature].values for feature in self.features],
            ]
        )

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _update_balance(self, price_diff: float, trade_volume: int):
        fees = abs(trade_volume) * self.fees_per_contract
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
        return self._get_space_spec(self.observation_space)

    def action_spec(self):
        return self._get_space_spec(self.action_space)

    def time_step_spec(self):
        return ts.time_step_spec(self.observation_spec())

    def _get_space_spec(self, space: gym.Space):
        return BoundedArraySpec(
            shape=space.shape, dtype=space.dtype, minimum=space.low, maximum=space.high
        )


class TFTradingEnvWrapper(TFEnvironment):
    def __init__(self, env: TradingEnvironment):
        super().__init__(
            time_step_spec=env.time_step_spec(), action_spec=env.action_spec()
        )

        self._env = env
        self._reset()

    def _current_time_step(self):
        return ts.TimeStep(
            step_type=self._current_step,
            reward=self._latest_reward,
            discount=self._discount,
            observation=self._env.observation,
        )

    def _reset(self):
        self._env.reset()
        self._discount = 1.0  # Currently not in use
        self._latest_reward = 0.0
        self._current_step = ts.StepType.FIRST

        return self._current_time_step()

    def _step(self, action):
        if self._current_step == ts.StepType.LAST:
            return self._reset()
        observation, reward, done, truncated, _info = self._env.step(action)
        self._current_step = ts.StepType.LAST if done or truncated else ts.StepType.MID

        return ts.TimeStep(
            step_type=self._current_step,
            reward=reward,
            discount=self._discount,
            observation=observation,
        )

    def render(self):
        self._env.render()


class PyTradingEnvWrapper(PyEnvironment):
    def __init__(self, env: TradingEnvironment, handle_auto_reset: bool = False):
        super().__init__(handle_auto_reset)

        self._env = env
        self._latest_info = {}
        self._reset()

    def observation_spec(self) -> types.NestedArraySpec:
        return self._env.observation_spec()

    def action_spec(self) -> types.NestedArraySpec:
        return self._env.action_spec()

    def time_step_spec(self) -> ts.TimeStep:
        return self._env.time_step_spec()

    def render(self, mode: Text = "rgb_array") -> Optional[types.NestedArray]:
        self._env.render()

    def get_info(self) -> types.NestedArray:
        def dict_to_nested_arrays(dictionary):
            if isinstance(dictionary, dict):
                nested_array = []
                for key, value in dictionary.items():
                    nested_value = dict_to_nested_arrays(value)
                    nested_array.append((key, nested_value))
                return nested_array
            else:
                return dictionary

        return dict_to_nested_arrays(self._latest_info)

    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        observation, reward, done, truncated, info = self._env.step(action)

        self._latest_info = info
        self._current_time_step = ts.TimeStep(
            step_type=ts.StepType.LAST if done or truncated else ts.StepType.MID,
            reward=reward,
            discount=self._discount,
            observation=observation,
        )

        return self._current_time_step

    def _reset(self) -> ts.TimeStep:
        observation = self._env.reset()
        self._discount = 1.0  # Currently not in use
        self._latest_info = {}
        self._current_time_step = ts.TimeStep(
            step_type=ts.StepType.FIRST,
            reward=0.0,
            discount=0.0,
            observation=observation,
        )

        return self.current_time_step()
