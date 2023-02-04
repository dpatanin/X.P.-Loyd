import pandas as pd
import numpy as np
import os
import pickle
import tensorflow as tf
import random

from tqdm import tqdm
from collections import deque


class FreeLaborTrader:
    def __init__(self, state_size, action_space=4, window_size=60):
        self.state_size = state_size
        self.action_space = action_space
        self.window_size = window_size
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995

        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()
        # TODO: Make LSTM work, input shape differs from state shape
        # model.add(
        #     tf.keras.layers.LSTM(32, input_shape=(self.window_size, self.state_size))
        # )
        model.add(
            tf.keras.layers.Dense(
                units=32, activation="relu", input_dim=self.state_size
            )
        )
        model.add(tf.keras.layers.Dense(units=128, activation="relu"))
        model.add(tf.keras.layers.Dense(units=self.action_space, activation="linear"))
        model.compile(loss="mean_squared_error", optimizer="adam")

        return model

    def trade(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)

        actions = self.model.predict(state)
        return np.argmax(actions[0])

    def batch_train(self, batch_size):
        batch = [
            self.memory[i]
            for i in range(len(self.memory) - batch_size + 1, len(self.memory))
        ]

        for state, action, reward, next_state, done in batch:
            # Reward if agent is in terminal state
            reward = reward

            # TODO: Do we need diminished returns?
            if not done:
                reward = reward + self.gamma * np.amax(
                    self.model.predict(next_state)[0]
                )

            target = self.model.predict(state)
            target[0][action] = reward

            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, filename):
        with open(filename, "rb") as f:
            self.model = pickle.load(f)


def load_data(file_path: str):
    assert os.path.exists(file_path), f"{file_path} does not exist."
    data = pd.read_csv(file_path)

    # Check for the presence of specific columns
    # TODO: DateTime mandatory if data processed by using datetime
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    assert set(required_columns).issubset(
        data.columns
    ), f"File {file_path} does not contain the required columns: {required_columns}"

    return data


class State:
    def __init__(
        self,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: int,
        balance=10000.00,
        entry_price=0.00,
        contracts=0,
    ):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.balance = balance
        self.entry_price = entry_price
        self.contracts = contracts

    def enter_long(self, entry_price: float, contracts: int, price_per_contract: float):
        assert not self.has_position(), "Exit current position first."
        self.entry_price = entry_price
        self.balance -= contracts * price_per_contract
        self.contracts = contracts

    def enter_short(
        self, entry_price: float, contracts: int, price_per_contract: float
    ):
        assert not self.has_position(), "Exit current position first."
        self.entry_price = entry_price
        self.balance += contracts * price_per_contract
        self.contracts = contracts

    def exit_position(self, exit_price: float, price_per_contract: float) -> float:
        assert self.has_position(), "No position to exit."
        profit = (exit_price - self.entry_price) * self.contracts * price_per_contract
        self.balance += profit
        self.entry_price = 0
        self.contracts = 0

        return profit

    def has_position(self):
        """
        return: 1 for long position, -1 for short position, 0 for no position
        """
        if self.contracts > 0:
            return 1
        elif self.contracts < 0:
            return -1
        else:
            return 0

    def rep_position(self):
        if not self.has_position():
            return "No position."

        position = self.has_position()
        return f"{'Long' if position == 1 else 'Short'} position: {self.contracts} contracts entered at {self.entry_price}."

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": [self.open],
                "high": [self.high],
                "low": [self.low],
                "close": [self.close],
                "volume": [self.volume],
                "balance": [self.balance],
                "entry_price": [self.entry_price],
                "contracts": [self.contracts],
            }
        )


def state_creator(data: pd.DataFrame, timestep: int, state: State = None):
    new_data = data.iloc[[timestep]].to_dict()
    new_state = State(
        new_data["Open"][timestep],
        new_data["High"][timestep],
        new_data["Low"][timestep],
        new_data["Close"][timestep],
        new_data["Volume"][timestep],
    )

    if state:
        new_state.balance = state.balance
        new_state.entry_price = state.entry_price
        new_state.contracts = state.contracts

    return new_state


data = load_data("data/ES_futures_sample/ES_continuous_1min_sample.csv")
episodes = 100
batch_size = 32
data_samples = len(data) - 1
tick_size = 0.25
tick_value = 12.50
initial_balance = 10000

trader = FreeLaborTrader(state_size=8)
trader.model.summary()
trader.save_model("models/financial_model_tensorflow_v0.1.pkl")


for episode in range(1, episodes + 1):
    print(f"Episode: {episode}/{episodes}")
    state = state_creator(data, 0)

    print(state.to_df().to_numpy())

    # tqdm is used for visualization
    for t in tqdm(range(data_samples)):
        action = trader.trade(state.to_df().to_numpy())
        next_state = state_creator(data, t + 1, state)
        reward = 0

        if action == 1 and not state.has_position():  # Buying; enter long position
            # TODO: Make possible to buy multiple contracts based on current balance
            next_state.enter_long(state.close, 1, tick_value)
            # print("FreeLaborTrader entered position:", state.rep_position())

        elif action == 2 and not state.has_position():  # Selling; enter short position
            # TODO: Make possible to sell short multiple contracts based on current balance
            next_state.enter_short(state.close, 1, tick_value)
            # print("FreeLaborTrader entered position:", state.rep_position())

        elif action == 3 and state.has_position():  # Exit; close position
            # TODO: Calculate actual profit
            # TODO: Calculate reward based on position exit
            profit = next_state.exit_position(state.close, tick_value)
            # print("FreeLaborTrader exited position with profit: ", profit)
            reward = profit

        done = t == data_samples - 1
        if done:
            # Consequences for braking restrictions
            reward = (
                -1000000000000000
                if state.has_position() or state.balance < 0
                else reward
            )
            print("########################")
            print(f"TOTAL PROFIT: {state.balance - initial_balance}")
            print("########################")

        trader.memory.append(
            (
                state.to_df().to_numpy(),
                action,
                reward,
                next_state.to_df().to_numpy(),
                done,
            )
        )

        state = next_state

        if len(trader.memory) > batch_size:
            trader.batch_train(batch_size)

    # Save the model every 10 episodes
    if episode % 10 == 0:
        trader.save_model(f"models/v0.1_ep{episode}.pkl")
