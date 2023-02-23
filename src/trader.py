from src.experience_replay import ExperienceReplayBuffer
from src.state import State
import numpy as np
import tensorflow as tf
import random


class FreeLaborTrader:
    def __init__(
        self,
        sequence_length: int,
        batch_size: int,
        num_features: int,
        action_space: int = 4,
        update_freq: int = 1,
    ):
        self.num_features = num_features
        self.action_space = action_space
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.memory = ExperienceReplayBuffer(2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995
        self.target_update_freq = update_freq
        self.target_update_cd = update_freq

        self.optimizer = tf.keras.optimizers.Adamax()
        self.model = self.build_model()
        self.target_model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()
        # Add a Dense layer as input layer
        model.add(
            tf.keras.layers.Conv1D(
                filters=32,
                kernel_size=3,
                input_shape=(self.sequence_length, self.num_features),
                activation="relu",
            )
        )

        # Add a LSTM layer
        model.add(tf.keras.layers.LSTM(units=64, return_sequences=True))

        # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
        model.add(tf.keras.layers.GRU(units=256, return_sequences=True))

        # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
        model.add(tf.keras.layers.SimpleRNN(units=128))

        # Add a Dense layer for the output
        model.add(tf.keras.layers.Dense(self.action_space, activation="linear"))

        model.compile(loss="mean_squared_error", optimizer=self.optimizer)

        return model

    def predict_action(self, states: list["State"]):
        """
        Given a batch of states, returns an action based on the current policy.

        Args:
            states: A list of states of size `batch_size`.

        Returns:
            An integer representing the action to take.
        """
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)

        # TODO: which action to choose?
        actions = self.model.predict(self.__transform_states(states))
        return np.argmax(actions[0])

    def batch_train(self):
        self.target_update_cd -= 1

        (states, actions, rewards, next_states, dones) = self.memory.sample(
            self.batch_size
        )

        states = self.__transform_states(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = self.__transform_states(next_states)
        dones = np.array(dones)

        # Combine states and goals
        # states = np.concatenate((states, achieved_goals), axis=-1)
        # next_states = np.concatenate((next_states, desired_goals), axis=-1)

        # Convert the dones list to a binary mask
        masks = 1 - dones
        masks = masks.astype(np.float32)

        # Update the model parameters
        with tf.GradientTape() as tape:
            # Compute Q-values for each state using the model
            q_values = self.model(states)
            q_values = tf.reduce_sum(
                q_values * tf.one_hot(actions, self.action_space), axis=1
            )

            # Compute the actual Q-values
            target_q_values = rewards + self.gamma * tf.reduce_max(
                self.target_model(next_states), axis=1
            ) * masks * tf.cast(
                tf.reduce_max(self.model(next_states), axis=1)
                == tf.reduce_max(self.target_model(next_states), axis=1),
                tf.float32,
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

        if self.target_update_cd == 0:
            self.update_target_model()
            self.target_update_cd = self.target_update_freq

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def __transform_states(self, states: list["State"]):
        return np.array([s.to_numpy() for s in states])
