import gym
import numpy as np
from gym import spaces


class TradingEnvironment(gym.Env):
    def __init__(
        self,
        lstm_forecast,
        ar_forecast,
        gru_forecast,
        close_prices,
        balance=1000,
        tick_ratio=12.5 / 0.25,
    ):
        super(TradingEnvironment, self).__init__()

        self.observation = self.init_observation = {
            "lstm_forecast": lstm_forecast,
            "ar_forecast": ar_forecast,
            "gru_forecast": gru_forecast,
            "close_prices": close_prices,
            "position": 0,  # Initialize position to neutral (no position); 1 = Long; 2 = Short
            "balance": balance,
        }
        self.entry_price = 0
        self.tick_ratio = tick_ratio

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict(
            {
                "lstm_forecast": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=lstm_forecast.shape,
                    dtype=np.float32,
                ),
                "ar_forecast": spaces.Box(
                    low=-np.inf, high=np.inf, shape=ar_forecast.shape, dtype=np.float32
                ),
                "gru_forecast": spaces.Box(
                    low=-np.inf, high=np.inf, shape=gru_forecast.shape, dtype=np.float32
                ),
                "close_prices": spaces.Box(
                    low=-np.inf, high=np.inf, shape=close_prices.shape, dtype=np.float32
                ),
                "position": spaces.Discrete(3),
            }
        )

    def reset(self):
        return self.init_observation

    def step(self, action):
        previous_position = self.observation["position"]
        self.observation["position"] = action
        next_observation = self.observation.copy()

        reward = self.calculate_reward(
            previous_position, action, self.observation["close_prices"][-1]
        )
        next_observation["balance"] += reward
        done = next_observation["balance"] <= 0

        return next_observation, reward, done

    def calculate_reward(
        self, previous_position, current_position, current_close_price
    ):
        if current_position != previous_position:
            if previous_position == 1:  # Long position
                reward = (current_close_price - self.entry_price) * self.tick_ratio
            elif previous_position == 2:  # Short position
                reward = (self.entry_price - current_close_price) * self.tick_ratio
            else:  # No position
                reward = 0
            self.entry_price = current_close_price
        else:
            reward = 0
        return reward
