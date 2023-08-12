from typing import TypedDict

import keras
import numpy as np
import pandas as pd
import tensorflow as tf


class EnsembleConfig(TypedDict):
    lstm_model_paths_and_columns: list[tuple[str, list[str]]]
    ar_model_path: str
    ar_columns: list[str]
    gru_model_path: str
    gru_columns: list[str]
    lstm_window: float
    ar_window: float
    gru_window: float


class Ensemble:
    """
    Ensembles DNN models.
    `window` overwrites unspecified windows.
    """

    def __init__(self, config: EnsembleConfig):
        self.max_window_size = max(
            config["lstm_window"],
            config["ar_window"],
            config["gru_window"],
        )
        self.lstm_window = config["lstm_window"]
        self.ar_window = config["ar_window"]
        self.gru_window = config["gru_window"]

        self.lstm_models_and_columns = [
            (self._load(model), columns)
            for model, columns in config["lstm_model_paths_and_columns"]
        ]
        self.ar_model = self._load(config["ar_model_path"])
        self.gru_model = self._load(config["gru_model_path"])

        self.ar_columns = config["ar_columns"]
        self.gru_columns = config["gru_columns"]

    def forecast(self, data: pd.DataFrame):
        assert len(data) <= self.max_window_size, "Data sequence too short."

        lstm_prediction = np.array(
            [
                model([data[columns].iloc[-self.lstm_window :]])[0].cpu().numpy()
                for model, columns in self.lstm_models_and_columns
            ]
        ).mean(0)

        ar_data = data[self.ar_columns].iloc[-self.ar_window :]
        ar_prediction = self.ar_model([ar_data])[0].cpu().numpy().flatten()

        gru_data = data[self.gru_columns].iloc[-self.gru_window :]
        gru_prediction = self.gru_model([gru_data])[0].cpu().numpy()

        return {"lstm": lstm_prediction, "ar": ar_prediction, "gru": gru_prediction}

    def _load(self, path) -> keras.Sequential:
        return tf.saved_model.load(path)
