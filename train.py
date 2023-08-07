import shutil
from datetime import datetime

import keras
import pandas as pd
import yaml
from yaml.loader import FullLoader

from lib.autoregressive import Autoregressive
from lib.data_processor import DataProcessor
from lib.window_generator import WindowGenerator

with open("config.yaml") as f:
    config = yaml.load(f, Loader=FullLoader)

now = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
model_dir = f"{config['model_directory']}/{config['description']}_{now}/"

dp = DataProcessor(config["data_src"])
wg = WindowGenerator(
    input_width=config["sequence_length"],
    label_width=config["prediction_length"],
    data_columns=dp.train_df.columns,
    label_columns=["close"],
    batch_size=config["batch_size"],
)


def compile_and_fit(
    model,
    name: str,
    train: pd.DataFrame = None,
    val: pd.DataFrame = None,
    test: pd.DataFrame = None,
):
    tb_callback = keras.callbacks.TensorBoard(
        log_dir=f"{config['log_dir']}/{config['description']}_{now}/{name}_{now}",
        update_freq=100,
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, mode="min"
    )
    optimizer = keras.optimizers.Adamax()
    loss = keras.losses.MeanSquaredError()
    metrics = [keras.metrics.MeanAbsoluteError()]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(
        wg.make_dataset(train if train is not None else dp.train_df),
        epochs=config["episodes"],
        validation_data=wg.make_dataset(val if val is not None else dp.val_df),
        callbacks=[tb_callback, early_stopping],
    )
    model.evaluate(
        wg.make_dataset(test if test is not None else dp.test_df),
        verbose=0,
        callbacks=[tb_callback],
    )

    model.save(f"{model_dir}{name}/", save_format="tf")
    shutil.copy2("config.yaml", f"{model_dir}{name}/")


def single_shot():
    return keras.Sequential(
        [
            keras.layers.LSTM(64, return_sequences=False),
            keras.layers.Dense(
                config["prediction_length"],
                kernel_initializer=keras.initializers.zeros(),
            ),
        ]
    )


lstm_close = single_shot()
compile_and_fit(model=lstm_close, name="lstm_close")

wg.compute_indices(shift=config["prediction_length"] + 1, label_columns=["open"])
lstm_open = single_shot()
compile_and_fit(model=lstm_open, name="lstm_open")

wg.compute_indices(
    shift=config["prediction_length"],
    data_columns=dp.train_df[["close"]].columns,
    label_columns=["close"],
)
ar_model = Autoregressive(32, config["prediction_length"])
compile_and_fit(
    model=ar_model,
    name="autoregressive",
    train=dp.train_df[["close"]],
    val=dp.val_df[["close"]],
    test=dp.test_df[["close"]],
)
