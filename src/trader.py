import numpy as np
import tensorflow as tf
import random

from collections import deque


class FreeLaborTrader:
    def __init__(self, state_size, action_space=4):
        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995

        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Dense(
                units=32, activation="relu", input_dim=self.state_size
            )
        )
        model.add(tf.keras.layers.Dense(units=128, activation="relu"))
        model.add(tf.keras.layers.Dense(units=self.action_space, activation="linear"))
        model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adamax())

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
