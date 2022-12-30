import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr

from tqdm import tqdm_notebook, tqdm
from collections import deque

import yfinance as yf
yf.pdr_override()

class AI_Trader():

    def __init__(self, state_size, action_space=3, model_name="AITrader"):


        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen = 2000)
        self.inventory = []
        self.model_name = model_name
        self.model = self.model_builder()

        self.gammsa = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995


    def model_builder(self):

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Dense(
            units=32, activation='relu', input_dim=self.state_size))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dense(
            units=self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

        return model

    def trade(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)

        actions = self.model.predict(actions[0])

    def batch_train(self, batch_size):

        batch = [
            self.memory[i]
            for i in range(len(self.memory) - batch_size + 1, len(self.memory))
        ]
        
        for state, action, reward, next_state, done in batch:
            reward = reward
            if not done:
                reward = reward + self.gamma * \
                        np.amax(self.model.predict(next_state)[0])

            target = self.model.predict(state)
            target[0][action] = reward

            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

def sigmoid(x):
        return (1 + math.exp(-x))

def stocks_price_format(n):
    return "- # {0:2f}".format(abs(n)) if n < 0 else "$ {0:2f}".format(abs(n))

def state_creator(data, timestep, window_size):
    starting_id = timestep - window_size + 1

    windowed_data = (
        data[starting_id : timestep + 1]
        if starting_id >= 0
        else starting_id * [data[0]] + list(data[: timestep + 1])
    )
    state = [
        sigmoid(windowed_data[i + 1] - windowed_data[i])
        for i in range(window_size - 1)
    ]
    return np.array([state])

# Loading data
data = pdr.get_data_yahoo("AAPL", start="2017-01-01", end="2017-04-30")

window_size = 10
episodes = 1000

batch_size = 32
data_samples = len(data) - 1

trader = AI_Trader(window_size)
trader.model.summary()


for episode in range(1, episodes + 1):

    print(f"Episode: {episode}/{episodes}")

    state = state_creator(data, 0, window_size + 1)

    total_profit = 0
    trader.inventory = []

    for t in tqdm(range(data_samples)):

        action = trader.trade(state)

        next_state = state_creator(data, t+1, window_size + 1)
        reward = 0

        if action == 1:  # Buying
          trader.inventory.append(data[t])
          print("AI Trader bought: ", stocks_price_format(data[t]))

        elif action == 2 and len(trader.inventory) > 0:  # Selling
          buy_price = trader.inventory.pop(0)

          reward = max(data[t] - buy_price, 0)
          total_profit += data[t] - buy_price
          print("AI Trader sold: ", stocks_price_format(
              data[t]), " Profit: " + stocks_price_format(data[t] - buy_price))

        done = t == data_samples - 1
        trader.memory.append((state, action, reward, next_state, done))

        state = next_state

        if done:
            print("########################")
            print(f"TOTAL PROFIT: {total_profit}")
            print("########################")

        if len(trader.memory) > batch_size:
          trader.batch_train(batch_size)

    if episode % 10 == 0:
        trader.model.save(f"ai_trader_{episode}.h5")
