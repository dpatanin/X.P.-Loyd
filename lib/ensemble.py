import keras
import numpy as np
import pandas as pd


class Ensemble:
    """
    Ensembles DNN models.
    `window` overwrites unspecified windows.
    """

    def __init__(
        self,
        lstm_models_and_columns: list[tuple[keras.Sequential, list[str]]],
        ar_model: keras.Model,
        ar_columns: list[str],
        gru_model: keras.Sequential,
        gru_columns: list[str],
        window: float = None,
        lstm_window: float = None,
        ar_window: float = None,
        gru_window: float = None,
    ) -> None:
        self.max_window_size = max(
            lstm_window or 0, ar_window or 0, gru_window or 0, window or 0
        )
        self.lstm_window = lstm_window or window
        self.ar_window = ar_window or window
        self.gru_window = gru_window or window

        self.lstm_models_and_columns = lstm_models_and_columns
        self.ar_model = ar_model
        self.gru_model = gru_model

        self.ar_columns = ar_columns
        self.gru_columns = gru_columns

    def forecast(self, data: pd.DataFrame):
        assert len(data) <= self.max_window_size, "Data sequence too short."

        lstm_prediction = np.array(
            [
                model([data[columns].iloc[-self.lstm_window :]])[0]
                for model, columns in self.lstm_models_and_columns
            ]
        ).mean(0)

        ar_data = data[self.ar_columns].iloc[-self.ar_window :]
        ar_prediction = self.ar_model([ar_data])[0]

        gru_data = data[self.gru_columns].iloc[-self.gru_window :]
        gru_prediction = self.gru_model([gru_data])[0]

        return {"lstm": lstm_prediction, "ar": ar_prediction, "gru": gru_prediction}
