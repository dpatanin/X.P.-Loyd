import json
import os

import gymnasium as gym
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
        max_ticks_without_action=23 * 60,
        checkpoint_length: int = 23 * 60 * 31,
        env_state_dir: str = None,
    ):
        """
        If `episode_history` is not None, the latest checkpoint will be loaded from it and training continues at that point.
        """
        super(TradingEnvironment, self).__init__()

        self._df = df
        self._window_size = window_size
        self._features = features
        self._trade_volume = trade_volume

        self._max_ticks_without_action = max_ticks_without_action
        self._tick_ratio = tick_ratio
        self._fees_per_contract = fees_per_contract

        self.action_space = spaces.Discrete(3)  # 0 = No position; 1 = Long; 2 = Short

        obs_shape = (3 + window_size * len(features),)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape)

        self._initial_balance = balance
        self._ticker = TradingEnvTicker(
            start_tick=window_size,
            end_tick=len(self._df) - 1,
            checkpoint_length=checkpoint_length,
        )

        self._num_prev_eps = 0
        self.history: list[pd.DataFrame] = []
        self.reset()
        self.history.clear()

        if env_state_dir:
            self.load_env_state(env_state_dir)

    def reset(self):
        self._done = False
        self._truncated = False
        self._balance = self._initial_balance
        self._position = 0
        self.history.append(self._new_history_ep())
        self._ticker.reset_to_checkpoint()
        self._update_observation(0)

        info = self._get_info(0.00, 0.00)
        self._update_history(info)

        return self.observation, info

    def step(self, action: int):
        perform_action = action != self._position
        self._done = self._ticker.update(perform_action)

        current_close_price = self._df["close"].iloc[self._ticker.current_tick]
        profit, fees = (
            self._take_action(action, current_close_price)
            if perform_action
            else (0.0, 0.0)
        )
        self._position = action

        self._update_observation(current_close_price)
        reward = self._calculate_reward(profit, fees)

        info = self._get_info(profit, fees)
        self._update_history(info)

        if self._done:
            self._ticker.restart_from_beginning()

        return (self.observation, reward, self._done, self._truncated, info)

    def _take_action(self, action: int, current_close_price: float):
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
        next_tick = self._ticker.current_tick + 1
        next_window = self._df[(next_tick - self._window_size) : next_tick]
        entry_price_diff = (
            current_close_price - self._entry_price if self._position != 0 else 0
        )

        self.observation = np.concatenate(
            [
                [self._balance - self._initial_balance],
                [self._position],
                [entry_price_diff],
                *[next_window[feature].values for feature in self._features],
            ],
            dtype=self.observation_space.dtype,
        )

    def _update_balance(self, price_diff: float):
        fees = self._trade_volume * self._fees_per_contract
        profit = self._trade_volume * price_diff * self._tick_ratio

        self._balance += profit - fees
        return (profit, fees)

    def _calculate_reward(self, profit: float, fees: float):
        # Calculate the composite reward
        laziness_punishment = max(
            self._ticker.ticks_since_last_action - self._max_ticks_without_action, 0
        )

        return (
            profit - fees + self._balance - self._initial_balance - laziness_punishment
        )

    def _get_info(self, profit, fees):
        return {
            "profit": profit,
            "fees": fees,
            "balance": self._balance,
            "position": self._position,
        }

    def _new_history_ep(self):
        return pd.DataFrame(columns=self._get_info(0, 0).keys())

    def _update_history(self, info: dict):
        self.history[-1].loc[self._ticker.current_tick] = info.values()

    def save_history(self, dir: str):
        for id, df in enumerate(self.history):
            df.to_csv(f"{dir}/ep{id + 1 + self._num_prev_eps}.csv")

    def save_env_state(self, dir: str):
        state = {
            "balance": self._balance,
            "position": self._position,
            "entryPrice": self._entry_price,
            "currentTick": self._ticker.current_tick,
            "checkpoint": self._ticker.checkpoint,
            "ticksSinceLastAction": self._ticker.ticks_since_last_action,
            "done": self._done,
            "truncated": self._truncated,
            "episodes": len(self.history) + self._num_prev_eps,
        }
        with open(f"{dir}/env_save.json", "w") as json_file:
            json.dump(state, json_file, default=serialize_numpy)

    def load_env_state(self, dir: str):
        with open(f"{dir}/env_save.json", "r") as json_file:
            state = json.load(json_file)
            self._balance = state["balance"]
            self._position = state["position"]
            self._entry_price = state["entryPrice"]
            self._ticker.current_tick = state["currentTick"]
            self._ticker.checkpoint = state["checkpoint"]
            self._ticker.ticks_since_last_action = state["ticksSinceLastAction"]
            self._done = state["done"]
            self._truncated = state["truncated"]
            self._num_prev_eps = state["episodes"]

        self.history = [self._new_history_ep()]
        self._update_observation(0)
        self._update_history(self._get_info(0.00, 0.00))

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


class TradingEnvTicker:
    def __init__(
        self,
        start_tick: int,
        end_tick: int,
        checkpoint_length: int,
        checkpoint: int = None,
    ) -> None:
        self.checkpoint = checkpoint or start_tick
        self._checkpoint_length = checkpoint_length

        self.current_tick = checkpoint or start_tick
        self.ticks_since_last_action = 0
        self._start_tick = start_tick
        self._end_tick = end_tick

    def update(self, action_taken: bool = False):
        self.current_tick += 1

        self.ticks_since_last_action += (
            -self.ticks_since_last_action if action_taken else 1
        )

        if self.current_tick >= self.checkpoint + self._checkpoint_length:
            self.checkpoint = self.current_tick

        return self.current_tick == self._end_tick

    def reset_to_checkpoint(self):
        self.current_tick = self.checkpoint
        self.ticks_since_last_action = 0

    def restart_from_beginning(self):
        self.current_tick = self.checkpoint = self._start_tick
        self.ticks_since_last_action = 0


class TFPyTradingEnvWrapper(TFPyEnvironment):
    def __init__(self, env: TradingEnvironment):
        py_env = PyTradingEnvWrapper(env)
        super().__init__(py_env)

        self.save = py_env.save
        self.reset_checkpoints = py_env.reset_checkpoints


class PyTradingEnvWrapper(PyEnvironment):
    def __init__(self, env: TradingEnvironment):
        super().__init__()

        self._env = env
        self._latest_info = {}
        self._discount = np.array(1.0, dtype=np.float32)  # Currently not in use
        self._current_time_step = ts.TimeStep(
            step_type=ts.StepType.FIRST,
            reward=np.array(0.0, dtype=np.float32),
            discount=self._discount,
            observation=self._env.observation,
        )

    def observation_spec(self) -> types.NestedArraySpec:
        return self._env.observation_spec()

    def action_spec(self) -> types.NestedArraySpec:
        return self._env.action_spec()

    def time_step_spec(self) -> ts.TimeStep:
        return self._env.time_step_spec()

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

    def reset_checkpoints(self):
        self._env._ticker.restart_from_beginning()

    def save(self, dir: str):
        os.makedirs("/".join(dir.split("/")[:-1]), exist_ok=True)
        self._env.save_history(dir)
        self._env.save_env_state(dir)


def serialize_numpy(obj):
    if isinstance(obj, (np.generic, np.ndarray)):
        return obj.item()  # Convert numpy scalar to a Python scalar
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
