import numpy as np
import pandas as pd
import tensorflow as tf


class WindowGenerator:
    """
    This class can:

    Handle the indexes and offsets as shown in the diagrams above.
    Split windows of features into (features, labels) pairs.
    Plot the content of the resulting windows.
    Efficiently generate batches of these windows from the training, evaluation, and test data, using tf.data.Datasets.
    """

    def __init__(
        self,
        input_width: int,
        label_width: int,
        data_columns: pd.Index,
        batch_size: int,
        shift: int = None,
        label_columns: list[str] = None,
    ):
        self.batch_size = batch_size
        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }
        self.column_indices = {name: i for i, name in enumerate(data_columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift or label_width

        self.total_window_size = input_width + self.shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [
                    labels[:, :, self.column_indices[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data: pd.DataFrame):
        selected_data = data[list(self.column_indices.keys())]
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=selected_data.to_numpy(dtype=np.float32),
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size,
        )

        ds = ds.map(self.split_window)

        return ds

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.label_columns}",
            ]
        )
