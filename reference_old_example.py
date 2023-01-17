import math
import random
import numpy as np
import tensorflow as tf
from pandas_datareader import data as pdr

from tqdm import tqdm
from collections import deque

import yfinance as yf

yf.pdr_override()

tf.debugging.set_log_device_placement(True)

class AI_Trader:
    def __init__(
        self, state_size, action_space=3, model_name="AITrader"
    ):  # * Stay, Buy, Sell

        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen=2000)
        self.inventory = []
        self.model_name = model_name

        # * Define hyperparameter
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995

        # * Call a function to build a model trough this class constructor
        # * More parameters could be utilized to programmatically define network size (layers and neurons)
        self.model = self.model_builder()

    def model_builder(self):

        model = tf.keras.models.Sequential()

        model.add(
            tf.keras.layers.Dense(
                units=32, activation="relu", input_dim=self.state_size
            )
        )
        model.add(tf.keras.layers.Dense(units=64, activation="relu"))
        model.add(tf.keras.layers.Dense(units=128, activation="relu"))
        model.add(tf.keras.layers.Dense(units=self.action_space, activation="linear"))
        model.compile(
            loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
        )

        return model

    # * Trade function that takes state as an input and returns an action
    # * to perform in particular state
    def trade(self, state):

        # * Should we perform a random generated action or action defined in model?

        # * If value from our random generator is smaller or equal to our epsilon
        # * then we will return a random action from action_space [0-3)
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)

        # * If our random is greater than epsilon then we will use model to perform action
        actions = self.model.predict(state)
        # * return only a one number defining an action (#Stay - 0 , Buy - 1, Sell - 2)
        # * that has maximum probability
        return np.argmax(actions[0])

    def batch_train(self, batch_size):

        # * Iterate in memory, we do not want to randomly select data as we are dealing with
        # * time constraint data. We will always sample from the end of memory size of batch
        batch = [
            self.memory[i]
            for i in range(len(self.memory) - batch_size + 1, len(self.memory))
        ]

        # * Iterate trough batch of data and train the model for each sample from batch
        # * Order of variables in for loop is important
        for state, action, reward, next_state, done in batch:
            # * Reward if agent is in terminal state
            reward = reward

            # * Check that agent is not in terminal state
            # * If not in terminal state calculate reward for actions that could be played
            if not done:
                # * Discounted total reward:
                reward = reward + self.gamma * np.amax(
                    self.model.predict(next_state)[0]
                )

            # * Target variable that is predicted by the model (action)
            target = self.model.predict(state)
            target[0][action] = reward

            self.model.fit(state, target, epochs=1, verbose=0)

        # * We will decrease epsilon parameter that is 1 as defined in __init__  so
        # * so we can stop performing random actions at some point
        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay


# * Usually used at the end of a network for binary classification
# * It changes range of input to scale of [0,1]
# * So we can normalize input data for comparison day by day if they are on different scale
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def stocks_price_format(n):
    return "- # {0:2f}".format(abs(n)) if n < 0 else "$ {0:2f}".format(abs(n))


# * Data -> dataset to predict from, gathered by dataset_loader()
# * Timestep -> Day in the dataset that we want to predict for [0:datalength]
# * Window_size -> how many days in past we want to use to predict current status[1:datalength]
# * Try different setup to see what creates best fit
def state_creator(data, timestep, window_size):

    # * starting day of our state
    starting_id = timestep - window_size + 1

    windowed_data = (
        data[starting_id : timestep + 1]
        if starting_id >= 0
        else -starting_id * [data[0]] + list(data[: timestep + 1])
    )

    # * Iterate trough whole windowed_data minus current state (-1)
    # * Normalize the difference from current day and the next day
    # * Because the prices can be very different and we want them on same scale
    state = [
        sigmoid(windowed_data[i + 1] - windowed_data[i]) for i in range(window_size - 1)
    ]
    return np.array([state])


# * Loading data (from Apple)
data = pdr.get_data_yahoo("AAPL", start="2022-01-01", end="2022-12-31")["Adj Close"]

window_size = 10
episodes = 100

batch_size = 32
data_samples = len(data) - 1  # discard last value, that we will predict on

trader = AI_Trader(window_size)
trader.model.summary()

# * Check for GPU utilization
device_name = tf.test.gpu_device_name()
if not device_name:
    raise SystemError("GPU device not found")
print(f"Found GPU at: {device_name}")

for episode in range(1, episodes + 1):

    # * To keep track of training process
    # * .format populates {} with variables in .format(x,y)
    print(f"Episode: {episode}/{episodes}")

    # * Create state
    # * second parameter is timestep = 0
    state = state_creator(data, 0, window_size + 1)

    total_profit = 0
    # * Empty inventory before starting episode
    trader.inventory = []

    # * One timestep is one day so number of timesteps we have represent data we have
    # * tqdm is used for visualization
    for t in tqdm(range(data_samples)):

        # * First we will access action that is going to be taken by model
        action = trader.trade(state)

        # * Use action to get to next state(t+)
        next_state = state_creator(data, t + 1, window_size + 1)
        # * As we did not calculate anything up to this point reward is 0
        reward = 0

        if action == 1:  # Buying
            # * Put bought stock to inventory to trade with
            trader.inventory.append(data[t])
            print("AI Trader bought: ", stocks_price_format(data[t]))

        # * To sell we need to have something in inventory
        elif action == 2 and len(trader.inventory) > 0:  # Selling
            # * Check buy price, pop removes first value from list
            buy_price = trader.inventory.pop(0)

            # * If we gain money (current price - buy price) we have reward
            # * if we lost money then reward is 0
            reward = max(data[t] - buy_price, 0)
            total_profit += data[t] - buy_price
            print(
                "AI Trader sold: ",
                stocks_price_format(data[t]),
                f" Profit: {stocks_price_format(data[t] - buy_price)}",
            )

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
