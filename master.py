import pandas as pd
import numpy as np
import os
import pickle
import tensorflow as tf
import random
from tqdm import tqdm


class FreeLaborTrader:
    def __init__(
        self, state_size, action_space=4, window_size=60, balance=10000, position=0
    ):
        self.state_size = state_size
        self.action_space = action_space
        self.window_size = window_size
        self.balance = balance
        self.position = position

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995

        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.LSTM(32, input_shape=(self.window_size, self.state_size))
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

            if not done:
                reward = next_state.balance - state.balance
                if state.balance - next_state.balance > state.balance:
                    reward = (
                        -1000000
                    )  # Add a large negative reward for breaking the "balance cannot turn negative" restriction
            elif next_state.contracts != 0:
                reward = (
                    -1000000
                )  # Add a large negative reward for breaking the "contracts must be 0 at the end of the day" restriction

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
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    assert set(required_columns).issubset(
        data.columns
    ), f"File {file_path} does not contain the required columns: {required_columns}"

    # data["diff"] = data["Close"] - data["Close"].shift(1)
    # data.dropna(inplace=True)

    return data


def state_creator(data, timestep, balance, position):
    state = data[timestep]
    state["Balance"] = balance
    state["Position"] = position
    return np.array([state])


data = load_data("data/ES_futures_sample/ES_continuous_1min_sample.csv")
window_size = 10
episodes = 100
batch_size = 32
data_samples = len(data) - 1
tick_size = 0.25
tick_value = 12.50

trader = FreeLaborTrader(window_size)
trader.model.summary()
trader.save_model("models/financial_model_tensorflow_v0.1.pkl")


for episode in range(1, episodes + 1):
    print(f"Episode: {episode}/{episodes}")

    state = state_creator(data, 0, trader.balance, trader.position)

    total_profit = 0
    trader.position = 0 # Reset position before starting episode

    # tqdm is used for visualization
    for t in tqdm(range(data_samples)):
        action = trader.trade(state)

        if action == 1 and trader.position == 0:  # Buying; enter long position
            # TODO: Make possible to buy multiple contracts based on current balance
            trader.position += 1
            print("FreeLaborTrader entered long: ", trader.position)

        elif action == 2 and trader.position == 0:  # Selling; enter short position
            # TODO: Make possible to sell short multiple contracts based on current balance
            trader.position -= 1
            print("FreeLaborTrader entered short: ", trader.position)

        elif action == 3 and trader.position != 0: # Exit; sell bought or buy short, i.e. close current position
            # TODO: Calculate actual profit
            # TODO: Calculate reward based on position exit
            trader.position = 0
            print("FreeLaborTrader exited position: ", trader.position)

        next_state = state_creator(data, t + 1, window_size + 1)
        # * As we did not calculate anything up to this point reward is 0
        reward = 0
        
        # * if t is last sample in our dateset we are done
        # * we do not have any steps to perform in current episode
        done = t == data_samples - 1
        # * Append all data to trader-agent memory, experience buffer
        trader.memory.append((state, action, reward, next_state, done))

        # * change state to next state, so we are done with an episode
        state = next_state

        if done:
            print("########################")
            print(f"TOTAL PROFIT: {total_profit}")
            print("########################")

        # * Check if we have more information in our memory than batch size
        if len(trader.memory) > batch_size:
            trader.batch_train(batch_size)

    # * Save the model every 10 episodes
    if episode % 10 == 0:
        trader.model.save(f"ai_trader_{episode}_epochs.h5")