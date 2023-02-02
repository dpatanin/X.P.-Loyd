import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pickle
import tensorflow as tf


class FinancialModel:
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.data = None
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(32, input_shape=(self.window_size, 4)))
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss="mean_squared_error", optimizer="adam")

        return model

    def load_data(self, file_path: str):
        assert os.path.exists(file_path), f"{file_path} does not exist."
        self.data = pd.read_csv(file_path)

        # Check for the presence of specific columns
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        assert set(required_columns).issubset(
            self.data.columns
        ), f"File {file_path} does not contain the required columns: {required_columns}"

        self.data["diff"] = self.data["Close"] - self.data["Close"].shift(1)
        self.data.dropna(inplace=True)

    def create_windowed_dataset(self):
        self.X = []
        self.y = []
        for i in range(self.window_size, len(self.data)):
            self.X.append(
                self.data[i - self.window_size : i][["Open", "High", "Low", "Volume"]]
            )
            self.y.append(self.data["Close"][i])
        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def split_data(self, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        self.X_train = self.X_train.reshape(
            self.X_train.shape[0], self.window_size, self.X_train.shape[2]
        )
        self.X_test = self.X_test.reshape(
            self.X_test.shape[0], self.window_size, self.X_test.shape[2]
        )

    def train_model(self, epochs=100, batch_size=32, verbose=0):
        self.model.fit(self.X_train, self.y_train, batch_size, epochs, verbose)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        mse = tf.keras.losses.mean_squared_error(y_pred, self.y_test).numpy()
        print("Mean Squared Error: {:.2f}".format(mse.mean()))

    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, filename):
        with open(filename, "rb") as f:
            self.model = pickle.load(f)

    def fit_model(self):
        self.create_windowed_dataset()
        self.split_data()
        self.train_model()
        self.evaluate_model()

class State:
    def __init__(self, data, balance, contracts, price_per_contract):
        self.data = data
        self.balance = balance
        self.contracts = contracts
        self.price_per_contract = price_per_contract
    
    def __repr__(self):
        return f"State(balance={self.balance}, contracts={self.contracts})"

class Action:
    BUY = 0
    SELL = 1
    SIT = 2
    
    def __init__(self, action):
        self.action = action
    
    def __repr__(self):
        if self.action == self.BUY:
            return "Action(BUY)"
        elif self.action == self.SELL:
            return "Action(SELL)"
        else:
            return "Action(SIT)"

def reward_function(state, next_state):
    reward = next_state.balance - state.balance
    if state.balance - next_state.balance > state.balance:
        reward = -1000000  # Add a large negative reward for breaking the "balance cannot turn negative" restriction
    if next_state.contracts != 0:
        reward = -1000000  # Add a large negative reward for breaking the "contracts must be 0 at the end of the day" restriction
    return reward

def next_state_function(state, action):
    next_state = State(data=state.data, balance=state.balance, contracts=state.contracts, price_per_contract=state.price_per_contract)
    
    if action.action == Action.BUY:
        next_state.balance -= state.contracts * state.price_per_contract
        next_state.contracts += state.contracts
    elif action.action == Action.SELL:
        next_state.balance += state.contracts * state.price_per_contract
        next_state.contracts -= state.contracts
    return next_state


model = FinancialModel()
model.load_data("data/ES_futures_sample/ES_continuous_1min_sample.csv")
model.fit_model()
model.save_model("models/financial_model_tensorflow_v0.1.pkl")
