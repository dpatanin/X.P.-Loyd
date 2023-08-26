import json
import os
from typing import Optional, Text

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from tf_agents.environments import PyEnvironment, TFPyEnvironment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types


class TradingEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int,
        features: list[str],
        trade_volume=1,
        balance=10000.00,
        tick_ratio=12.5 / 0.25,
        fees_per_contract=0.00,
        streak_span=15,
        streak_bonus_max=10.00,
        streak_difficulty=1.00,
        max_ticks_without_action=23 * 60,
        trade_reward_weight=0.7,
        balance_change_weight=0.3,
        episode_history: list[dict] = None,
        keep_full_history=False,
        checkpoint_length: int = None,
        checkpoint_tick: int = None,
    ):
        """
        A gym environment simulating simple day trading. Requires price data: `["high", "low", "open", "close"]` to be present in the df.
        If `episode_history` is not None, the latest checkpoint will be loaded from it and training continues at that point.
        If `checkpoint_tick` is provided, the checkpoint will be loaded from there instead.
        `checkpoint_length` is required for making checkpoints.
        `episode_history` will only keep last entry (episode) of loaded file unless `keep_full_history` is set to `True`.
        `streak_bonus_max` is the upper bound of the streak bonus multiplier.
        `streak_span` is the durational threshold for streaks.
        """
        super(TradingEnvironment, self).__init__()

        self._df = df
        self._window_size = window_size
        self._features = features
        self._trade_volume = trade_volume

        self._streak_span = streak_span
        self._streak_bonus_max = streak_bonus_max
        self._streak_difficulty = streak_difficulty
        self._max_ticks_without_action = max_ticks_without_action
        self._tick_ratio = tick_ratio
        self._fees_per_contract = fees_per_contract

        self.action_space = spaces.Discrete(3)  # 0 = No position; 1 = Long; 2 = Short

        obs_shape = (3 + window_size * len(features),)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape)

        self._initial_balance = balance
        self._prices = df[["high", "low", "open", "close"]]
        self._trade_reward_weight = trade_reward_weight
        self._balance_change_weight = balance_change_weight

        self._checkpoint_length = checkpoint_length or len(df)
        if episode_history:
            self._episode_history = (
                episode_history if keep_full_history else episode_history[-1:]
            )
            self._checkpoint = episode_history[-1]["checkpoint"][-1]
            self._start_tick = (
                checkpoint_tick or self._checkpoint * checkpoint_length
                if self._checkpoint > 0
                else window_size
            )
        else:
            self._episode_history = []
            self._checkpoint = 0
            self._start_tick = self._window_size

        self._end_tick = len(self._df) - 1

        self.reset()

    def reset(self):
        self._done = False
        self._truncated = False
        self._balance = self._initial_balance
        self._current_tick = self._start_tick
        self._streak = 1
        self._ticks_since_last_action = 0
        self._entry_price_diff = 0
        self._total_profit = 0
        self._total_fees = 0
        self._position = 0
        self._position_history = (self._window_size * [None]) + [self._position]
        self._first_rendering = True
        self.history = {}
        self._update_observation(0)

        info = self._get_info(0.00, 0.00)
        self._update_history(info)

        return self.observation, info

    def step(self, action: int):
        self._current_tick += 1
        self._ticks_since_last_action += 1
        self._done = self._current_tick == self._end_tick

        if self._current_tick >= self._start_tick + self._checkpoint_length:
            self._checkpoint += 1
            self._start_tick += self._checkpoint_length

        current_close_price = self._df["close"].iloc[self._current_tick]
        profit, fees = (
            self._take_action(action, current_close_price)
            if action != self._position
            else (0.0, 0.0)
        )
        self._position = action
        self._position_history.append(self._position)

        self._update_observation(current_close_price)
        self._update_streak(profit)
        self._update_weights(profit)
        reward = self._calculate_reward(profit, fees)

        info = self._get_info(profit, fees)
        self._update_history(info)

        if self._done or self._truncated:
            self._episode_history.append(self.history)

        return (self.observation, reward, self._done, self._truncated, info)

    def _take_action(self, action: int, current_close_price: float):
        self._ticks_since_last_action = 0
        price_diff = 0

        # Exit or switch (switch includes exit) -> Get price diff with correct sign
        if action == 0 or self._position != 0:
            if self._position == 1:
                price_diff = current_close_price - self._entry_price

            elif self._position == 2:
                price_diff = self._entry_price - current_close_price

            self._entry_price = 0

        # Enter or switch (switch includes enter) -> Set entry price
        if action != 0:
            self._entry_price = current_close_price

        profit, fees = self._update_balance(price_diff)
        self._truncated = self._balance <= 0

        return profit, fees

    def _update_observation(self, current_close_price: float):
        next_window = self._df[
            (self._current_tick - self._window_size + 1) : self._current_tick + 1
        ]
        entry_price_diff = (
            current_close_price - self._entry_price if self._position != 0 else 0
        )

        self.observation = np.concatenate(
            [
                [self._total_profit - self._total_fees],
                [self._position],
                [entry_price_diff],
                *[next_window[feature].values for feature in self._features],
            ],
            dtype=self.observation_space.dtype,
        )

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _update_balance(self, price_diff: float):
        fees = self._trade_volume * self._fees_per_contract
        self._total_fees += fees

        profit = self._trade_volume * price_diff * self._tick_ratio
        self._total_profit += profit

        self._balance += profit - fees
        return (profit, fees)

    def _update_streak(self, profit: float):
        profits = self.history["profit"][-self._streak_span + 1 :] + [profit]
        positive_count = sum(profit > 0 for profit in profits)

        x0 = self._trade_volume * self._tick_ratio * self._streak_difficulty
        k = self._streak_span + self._streak_difficulty - positive_count
        self._streak = 1 + self._streak_bonus_max / (1 + np.exp(-k * (profit - x0)))

    def _update_weights(self, profit: float):
        # When balance is low, momentary gains are less important; when balance rises, should not overshadow momentary gains
        # Calculate the ratio of trades to total net profit (+ 1 to avoid division by zero)
        trade_balance_ratio = abs(profit / (self._total_profit - self._total_fees + 1))

        # Adjust weights based on the trade balance ratio
        if trade_balance_ratio > 1.0:
            self._trade_reward_weight *= 0.95
            self._balance_change_weight *= 1.05
        else:
            self._trade_reward_weight *= 1.05
            self._balance_change_weight *= 0.95

        # Normalize the weights
        total_weight = self._trade_reward_weight + self._balance_change_weight
        self._trade_reward_weight /= total_weight
        self._balance_change_weight /= total_weight

    def _calculate_reward(self, profit: float, fees: float):
        # Calculate the composite reward
        streak = self._streak if profit >= 0 else 1
        laziness_punishment = (
            self._ticks_since_last_action - self._max_ticks_without_action
        )

        return (
            streak * (self._trade_reward_weight * (profit - fees))
            + (self._balance_change_weight * (self._balance - self._initial_balance))
            + laziness_punishment
        )

    def _get_info(self, profit, fees):
        return {
            "profit": profit,
            "total_profit": self._total_profit,
            "fees": fees,
            "total_fees": self._total_fees,
            "balance": self._balance,
            "position": self._position,
            "streak": self._streak,
            "checkpoint": self._checkpoint,
        }

    def render(self, mode="human"):
        def _plot_position(position, tick):
            if position == 0:
                color = "blue"
            elif position > 0:
                color = "green"
            elif position < 0:
                color = "red"
            else:
                color = None
            if color:
                plt.scatter(tick, self._prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self._prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)
        plt.pause(0.01)

    def render_all(self, mode="human"):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self._prices)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] < 0:
                short_ticks.append(tick)
            elif self._position_history[i] > 0:
                long_ticks.append(tick)

        plt.plot(short_ticks, self._prices[short_ticks], "ro")
        plt.plot(long_ticks, self._prices[long_ticks], "go")

    def close(self):
        plt.close()

    def save_rendering(self, filepath: str):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def observation_spec(self):
        return BoundedArraySpec(
            shape=self.observation_space.shape,
            dtype=self.observation_space.dtype,
            minimum=self.observation_space.low,
            maximum=self.observation_space.high,
            name="observation",
        )

    def action_spec(self):
        return BoundedArraySpec(
            shape=(), dtype=np.int64, name="action", minimum=0, maximum=2
        )

    def time_step_spec(self):
        return ts.time_step_spec(self.observation_spec())


class TFPyTradingEnvWrapper(TFPyEnvironment):
    def __init__(self, env: TradingEnvironment):
        py_env = PyTradingEnvWrapper(env)
        super().__init__(py_env)

        self.save_episode_history = py_env.save_episode_history


class PyTradingEnvWrapper(PyEnvironment):
    def __init__(self, env: TradingEnvironment):
        super().__init__()

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
        if self._current_time_step.is_last():
            return self._reset()
        observation, reward, done, truncated, info = self._env.step(action)

        self._latest_info = info
        self._current_time_step = ts.TimeStep(
            step_type=ts.StepType.LAST if done or truncated else ts.StepType.MID,
            reward=np.array(reward, dtype=np.float32),
            discount=self._discount,
            observation=observation,
        )

        return self._current_time_step

    def _reset(self) -> ts.TimeStep:
        observation, info = self._env.reset()
        self._discount = np.array(1.0, dtype=np.float32)  # Currently not in use
        self._latest_info = info
        self._current_time_step = ts.TimeStep(
            step_type=ts.StepType.FIRST,
            reward=np.array(0.0, dtype=np.float32),
            discount=self._discount,
            observation=observation,
        )

        return self.current_time_step()

    def save_episode_history(self, file_name: str):
        self._env._episode_history.append(self._env.history)
        os.makedirs("/".join(file_name.split("/")[:-1]), exist_ok=True)

        with open(f"{file_name}.json", "w") as json_file:
            json.dump(self._env._episode_history, json_file, default=serialize_numpy)


def serialize_numpy(obj):
    if isinstance(obj, (np.generic, np.ndarray)):
        return obj.item()  # Convert numpy scalar to a Python scalar
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
