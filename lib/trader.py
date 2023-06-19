from lib.experience_replay import HERBuffer
from lib.state import State
import numpy as np
import tensorflow as tf
import random


class FreeLaborTrader:
    """
    The trader defines the network and maintains a record of the transitions of size `memory_size`.
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
        memory_size=2000,
        update_freq=5,
        hindsight_reward_fac=1.00,
        gamma=0.95,
        epsilon=1.0,
        epsilon_final=0.01,
        epsilon_decay=0.995,
        learning_rate=0.01,
    ):
        self.num_features = num_features
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.memory = HERBuffer(memory_size, hindsight_reward_fac)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = update_freq
        self.target_update_cd = update_freq

        self.optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
        self.model = self.build_model()
        self.target_model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Conv1D(
                filters=32,
                kernel_size=3,
                input_shape=(self.sequence_length, self.num_features),
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

        return model

    def load(self, path: str) -> None:
        new_model = tf.keras.models.load_model(path)
        if self.model.get_config() == new_model.get_config():
            self.model = new_model
        else:
            raise AssertionError(
                "Loaded model differs from training setup.\n"
                + f"Expected: {self.model.get_config()}\n"
                + f"Loaded: {new_model.get_config()}"
            )

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

        (states, rewards, next_states, dones) = self.memory.sample(self.batch_size)
        # Convert the dones list to a binary mask; (1 for not done, 0 for done)
        masks = tf.cast(tf.logical_not(dones), dtype=tf.float32)

        # Update the model parameters
        with tf.GradientTape() as tape:
            # Access q values manually for GradientTape to record for backpropagation
            q_values = self.model(self.__transform_states(states))[:, 0]
            target_q_values = self.target_model(self.__transform_states(next_states))[
                :, 0
            ]

            # Compute the target Q-values following the bellmann equation
            target_q_values = (
                rewards + self.gamma * tf.reduce_max(target_q_values) * masks
            )

            loss = tf.reduce_mean(tf.square(q_values - target_q_values))
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
