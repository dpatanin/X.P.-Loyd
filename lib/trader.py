import random

import numpy as np
import tensorflow as tf

from lib.experience_replay import ExperienceReplayBuffer
from lib.state import State


class FreeLaborTrader:
    """
    The trader defines the network and maintains a record of the transitions.
    It's purpose is to represent the actor/agent in the environment, providing predictions, memorizing the outcomes
    and lastly training on those observations to improve the predictions.\n
    The trader uses two networks, one primary network is used to make the predictions,
    a secondary target network is used to ensure a more stable learning process, being updated every `update_freq` trainings.
    During training the trader uses an experience replay strategy, calculates the loss and updates the gradients.

    |`sequence_length`, `batch_size`, `num_features`: Mandatory configuration parameters.
    |`hindsight_reward_fac`: Weight for hindsight rewards.
    |`gamma`: Weight inside the loss functions.
    |`epsilon`, `epsilon_final`, `epsilon_decay`: Exploration parameters.
    """

    def __init__(
        self,
        sequence_length: int,
        batch_size: int,
        num_features: int,
        update_freq=5,
        gamma=0.95,
        epsilon=1.0,
        epsilon_final=0.01,
        epsilon_decay=0.995,
        learning_rate=0.01,
    ):
        self.batch_size = batch_size
        self.memory = ExperienceReplayBuffer(batch_size * 5)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = update_freq
        self.target_update_cd = update_freq

        self.optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
        self.model = self.__build_model(sequence_length, num_features)
        self.target_model = self.__build_model(sequence_length, num_features)

    def __build_model(self, input_dim1: int, input_dim2: int):
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Conv1D(
                filters=32,
                kernel_size=3,
                input_shape=(input_dim1, input_dim2),
                activation="relu",
            )
        )
        model.add(tf.keras.layers.LSTM(units=64, return_sequences=True))
        # Output of GRU: 3D tensor (batch_size, timesteps, 256)
        model.add(tf.keras.layers.GRU(units=256, return_sequences=True))
        # Output of SimpleRNN: 2D tensor (batch_size, 128)
        model.add(tf.keras.layers.SimpleRNN(units=128))
        # Single continuous output in range (-1, 1)
        model.add(tf.keras.layers.Dense(units=1, activation="tanh"))
        model.compile(loss="mean_squared_error", optimizer=self.optimizer)
        model.trainable = True

        return model

    def load(self, path: str) -> None:
        new_model = tf.keras.models.load_model(path)
        if self.model.get_config() != new_model.get_config():
            raise AssertionError(
                "Loaded model differs from training setup.\n"
                + f"Expected: {self.model.get_config()}\n"
                + f"Loaded: {new_model.get_config()}"
            )

        self.model = new_model
        self.target_model = tf.keras.models.load_model(path)  # Loading a copy

    def predict(self, states: list["State"]) -> list[float]:
        """
        Returns a list of q values of range (-1, 1).
        """
        q_values = self.model.predict_on_batch(self.__transform_states(states))[:, 0]
        random_values = tf.random.uniform(shape=q_values.shape, minval=-1, maxval=1)
        predictions = tf.where(random.random() <= self.epsilon, random_values, q_values)
        return predictions.numpy().tolist()

    def batch_train(self):
        self.target_update_cd -= 1

        memories = self.memory.sample(self.batch_size)
        states = [m.origin for m in memories]
        q_values = [m.q_value for m in memories]
        rewards = [m.reward for m in memories]
        next_states = [m.outcome for m in memories]
        dones = [m.done for m in memories]

        # Convert the dones list to a binary mask; (1 for not done, 0 for done)
        masks = tf.cast(tf.logical_not(dones), dtype=tf.float32)

        # Update the model parameters
        with tf.GradientTape() as tape:
            # Compute Q-values for the next states using the target model
            q_values_next = self.target_model(
                tf.convert_to_tensor(
                    self.__transform_states(next_states), dtype=tf.float32
                )
            )

            # Compute the target Q-values based on the Bellman equation
            # Q(s, a) = r + gamma * max(Q(s', a')) if the episode is not done
            # Q(s, a) = r if the episode is done (no future rewards)
            target_q_values = (
                rewards + self.gamma * tf.reduce_max(q_values_next, axis=1) * masks
            )

            # Compute the Q-values that were used to take the actions (stored in q_values)
            # These values are used as targets for the Q-values for the corresponding states
            target_q_values_for_actions = tf.convert_to_tensor(
                q_values, dtype=tf.float32
            )

            # Calculate the Mean Squared Error (MSE) loss between the target Q-values and the predicted Q-values
            loss = tf.reduce_mean(
                tf.square(target_q_values_for_actions - target_q_values)
            )

            # Compute gradients of the loss with respect to the model parameters
            gradients = tape.gradient(loss, self.model.trainable_variables)

            # Apply gradients to update the model parameters
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
