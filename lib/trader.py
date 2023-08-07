import tensorflow as tf


class PredictionModel(tf.keras.Sequential):
    def __init__(
        self,
        input_shape: tuple[float, float],
        learning_rate: float,
        num_output: int,
    ):
        super().__init__()
        self.num_output = num_output
        self.optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)

        self.add(
            tf.keras.layers.Conv1D(
                filters=32,
                kernel_size=3,
                input_shape=input_shape,
                activation="relu",
            )
        )
        self.add(tf.keras.layers.LSTM(units=64, return_sequences=True))
        self.add(tf.keras.layers.GRU(units=256, return_sequences=True))
        self.add(tf.keras.layers.SimpleRNN(units=128))
        self.add(tf.keras.layers.Dense(units=num_output))
        self.trainable = True

        self.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=self.optimizer,
            metrics=[tf.keras.metrics.MeanAbsoluteError()],
        )
