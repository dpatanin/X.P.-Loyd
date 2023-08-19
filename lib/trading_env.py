import json
from typing import Optional, Text

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from tf_agents.environments import PyEnvironment
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
        trade_limit=10,
        balance=10000.00,
        tick_ratio=12.5 / 0.25,
        fees_per_contract=0.00,
        streak_span=15,
        streak_bonus_max=10.00,
        episode_history: list[dict] = None,
        keep_full_history=False,
        checkpoint_length: int = None,
        checkpoint_tick: int = None,
    ):
        """
        A gym environment simulating simple day trading. Requires price data: `["high", "low", "open", "close"]` to be present in the df.
        If `episode_history` is not None, the latest checkpoint will be loaded from it and training continues at that point. (`checkpoint_length` is thus required)
        If `checkpoint_tick` is provided, the checkpoint will be loaded from there instead.
        `episode_history` will only keep last entry (episode) of loaded file unless `keep_full_history` is set to `True`.
        `streak_bonus_max` is the upper bound of the streak bonus multiplier.
        `streak_span` is the durational threshold for streaks.
        """
        super(TradingEnvironment, self).__init__()

        self._df = df
        self._window_size = window_size
        self._features = features
        self._trade_limit = trade_limit

        self.streak_span = streak_span
        self.streak_bonus_max = streak_bonus_max
        self.tick_ratio = tick_ratio
        self.fees_per_contract = fees_per_contract

        self.action_space = spaces.Box(low=-trade_limit, high=trade_limit, shape=(1,))

        obs_shape = (2 + window_size * len(features),)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape)

        self._init_balance = balance
        self._prices = df[["high", "low", "open", "close"]]

        if episode_history:
            self._episode_history = (
                episode_history if keep_full_history else episode_history[-1]
            )
            self._checkpoint = episode_history[-1]["checkpoint"][-1]
            self._start_tick = checkpoint_tick or self._checkpoint * checkpoint_length
        else:
            self._episode_history = []
            self._checkpoint = 0
            self._start_tick = self._window_size

        self._end_tick = len(self._df) - 1

        self.reset()

    def reset(self):
        self._done = False
        self._truncated = False
        self._balance = self._init_balance
        self._current_tick = self._start_tick
        self._streak = 1
        self._entry_price = 0
        self._total_profit = 0
        self._total_fees = 0
        self._position = 0
        self._position_history = (self._window_size * [None]) + [self._position]
        self._first_rendering = True
        self.history = {}
        self._update_observation()

        return self.observation, self._get_info(0.00, 0.00)

    def step(self, action: np.ndarray):
        self._current_tick += 1
        self._done = self._current_tick == self._end_tick
        if self._current_tick >= self._start_tick + self._checkpoint_length:
            self._checkpoint += 1
            self._start_tick += self._checkpoint_length

        trade_volume = action[0]
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
        self._update_streak(profit)
        reward = profit if profit <= 0 else self._streak * profit

        info = self._get_info(profit, fees)
        self._update_history(info)
        if self._done or self._truncated:
            self._episode_history.append(self.history)

        return (self.observation, reward, self._done, self._truncated, info)

    def _update_observation(self):
        next_window = self._df[
            (self._current_tick - self._window_size + 1) : self._current_tick + 1
        ]

        # Ensure order of inputs/outputs
        self.observation = np.concatenate(
            [
                [self._position],
                [self._balance],
                *[next_window[feature].values for feature in self._features],
            ],
            dtype=self.observation_space.dtype,
        )

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _update_balance(self, price_diff: float, trade_volume: int):
        fees = abs(trade_volume) * self.fees_per_contract
        self._total_fees += fees

        profit = trade_volume * price_diff * self.tick_ratio
        self._total_profit += profit

        self._balance += profit - fees
        return (profit, fees)

    def _update_streak(self, profit: float):
        profits = self.history["profit"][-self.streak_span + 1 :] + [profit]

        streak_bonus_half = (self.streak_bonus_max - 1) / 2
        positive_count = sum(profit > 0 for profit in profits)
        self._streak = 1 + streak_bonus_half * (positive_count / self.streak_span)

        severity_threshold = self._trade_limit * self.tick_ratio
        sum_positive = sum(value for value in reversed(profits) if value > 0)
        self._streak += (
            sum_positive / (self.streak_span / 3 * severity_threshold)
        ) * streak_bonus_half

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
        return self._get_space_spec(self.observation_space, "observation")

    def action_spec(self):
        return self._get_space_spec(self.action_space, "action")

    def time_step_spec(self):
        return ts.time_step_spec(self.observation_spec())

    def _get_space_spec(self, space: gym.Space, name: str):
        return BoundedArraySpec(
            shape=space.shape,
            dtype=space.dtype,
            minimum=space.low,
            maximum=space.high,
            name=name,
        )


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
        with open(f"{file_name}.json", "w") as json_file:
            json.dump(
                self._env._episode_history, json_file, default=self._serialize_numpy
            )

    def _serialize_numpy(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()  # Convert numpy scalar to a Python scalar
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
