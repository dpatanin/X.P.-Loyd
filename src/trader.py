from src.experience_replay import ExperienceReplayBuffer
import numpy as np
import tensorflow as tf
import random


class FreeLaborTrader:
    def __init__(self, state_size: int, action_space: int = 4):
        self.state_size = state_size
        self.action_space = action_space
        self.memory = ExperienceReplayBuffer(2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995

        self.optimizer = tf.keras.optimizers.Adamax()
        self.model = self.build_model()
        self.target_model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Dense(
                units=32, activation="relu", input_shape=(self.state_size,)
            )
        )
        model.add(tf.keras.layers.Dense(units=128, activation="relu"))
        model.add(tf.keras.layers.Dense(units=self.action_space, activation="linear"))
        model.compile(loss="mean_squared_error", optimizer=self.optimizer)

        return model

    def trade(self, state: np.ndarray):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)

        actions = self.model.predict(state)
        return np.argmax(actions[0])

    def batch_train(self, batch_size: int):
        (states, actions, rewards, next_states, dones) = self.memory.sample(batch_size)

        # Combine states and goals
        # states = np.concatenate((states, achieved_goals), axis=-1)
        # next_states = np.concatenate((next_states, desired_goals), axis=-1)

        # Convert the dones list to a binary mask
        masks = 1 - dones
        masks = masks.astype(np.float32)

        # Update the model parameters
        with tf.GradientTape() as tape:
            # Compute the predicted Q-values
            q_values = self.model(states)
            q_values = tf.reduce_sum(
                q_values * tf.one_hot(actions, self.action_space), axis=1
            )

            # Compute the actual Q-values
            target_q_values = (
                rewards
                + self.gamma * tf.reduce_max(self.model(next_states), axis=1) * masks
            )

            # Compute the loss
            loss = tf.reduce_mean(tf.square(q_values - target_q_values))

            # Compute the gradients
            gradients = tape.gradient(loss, self.model.trainable_variables)

            # Apply the gradients to update the model parameters
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
