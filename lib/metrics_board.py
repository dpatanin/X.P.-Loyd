import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class MetricsBoard:
    """
    Holds a record of captured metrics and plots them onto a tensorBoard.
    """

    def __init__(self, log_dir: str) -> None:
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)
        self.rewards = []
        self.losses = np.ndarray([])
        self.entropies = np.ndarray([])
        self.target_values = np.ndarray([])
        self.gradient_norms = np.ndarray([])

    def append_entropy(self, logits: tf.Tensor) -> None:
        policy_dist = tfp.distributions.Categorical(logits=logits)
        entropy = policy_dist.entropy()
        self.entropies = np.append(self.entropies, entropy.numpy())

    def append_target_values(self, logits: tf.Tensor) -> None:
        self.target_values = np.append(self.target_values, logits.numpy())

    def append_gradients(self, logits: tf.Tensor) -> None:
        self.gradient_norms = np.append(
            self.gradient_norms, [tf.norm(lg).numpy() for lg in logits]
        )

    def append_loss(self, logits: tf.Tensor) -> None:
        self.losses = np.append(self.losses, logits.numpy())

    def log_metrics(self, ep: int, exploration_rate: float, learning_rate: float):
        with self.train_summary_writer.as_default():
            tf.summary.scalar("avg_loss", self.losses.mean(), step=ep)
            tf.summary.scalar("avg_reward", np.mean(self.rewards), step=ep)
            tf.summary.scalar("avg_entropy", self.entropies.mean(), step=ep)
            tf.summary.scalar("avg_state_value", self.target_values.mean(), step=ep)
            tf.summary.scalar("avg_gradient_norm", self.gradient_norms.mean(), step=ep)
            tf.summary.scalar("exploration_rate", exploration_rate, step=ep)
            tf.summary.scalar("learning_rate", learning_rate, step=ep)

    def clear(self):
        self.rewards = []
        self.losses = np.ndarray([])
        self.entropies = np.ndarray([])
        self.target_values = np.ndarray([])
        self.gradient_norms = np.ndarray([])
