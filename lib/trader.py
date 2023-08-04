import tensorflow as tf


class PredictionModel(tf.keras.Model):
    def __init__(
        self,
        sequence_length: int,
        num_features: int,
        learning_rate: float,
        num_output: int,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)

        input_layer = tf.keras.Input(shape=(sequence_length, num_features))
        x = tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=3,
            activation="relu",
        )(input_layer)
        x = tf.keras.layers.GRU(units=256, return_sequences=True)(x)
        x = tf.keras.layers.SimpleRNN(units=128)(x)
        output_layer = tf.keras.layers.Dense(units=1)(x)

        self.inputs = input_layer
        self.lstm_cell = tf.keras.layers.LSTMCell(64)
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.outputs = output_layer
        self.dense = tf.keras.layers.Dense(units=1)
        self.trainable = True

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")
        self.compile(
            loss="mae", optimizer=self.optimizer, metrics=[tf.keras.metrics.Accuracy()]
        )

        self.num_output = num_output

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)

        predictions = [prediction]
        # Run the rest of the prediction steps.
        for _ in range(1, self.num_output):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state, training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions
