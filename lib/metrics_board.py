import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras.callbacks import TensorBoard


class MetricsBoard:
    """
    Holds a record of captured metrics and plots them onto a tensorBoard.
    """

    def __init__(self, log_dir: str) -> None:
        self.tensor_board = TensorBoard(log_dir=log_dir, write_graph=True)
        self.losses = []
        self.rewards = []
        self.entropies = []
        self.target_values = []
        self.gradient_norms = []

    def append_entropy(self, logits: tf.Tensor) -> None:
        policy_dist = tfp.distributions.Categorical(logits=logits)
        entropy = policy_dist.entropy()
        self.entropies.extend(entropy.numpy())

    def append_target_values(self, logits: tf.Tensor) -> None:
        self.entropies.extend(logits.numpy().flatten())

    def append_gradients(self, logits: tf.Tensor) -> None:
        self.gradient_norms.extend([tf.norm(lg).numpy() for lg in logits])

    def log_metrics(self, batch: int, exploration_rate: float, learning_rate: float):
        self.tensor_board.on_train_batch_end(
            batch,
            {
                "avg_loss": np.mean(self.losses),
                "avg_reward": np.mean(self.rewards),
                "avg_entropy": np.mean(self.entropies),
                "avg_state_value": np.mean(self.target_values),
                "avg_gradient_norm": np.mean(self.gradient_norms),
                "exploration_rate": exploration_rate,
                "learning_rate": learning_rate,
            },
        )

    def clear(self):
        self.losses = []
        self.rewards = []
        self.entropies = []
        self.target_values = []
        self.gradient_norms = []
