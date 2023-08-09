import shutil
from datetime import datetime

import keras
import yaml
from yaml.loader import FullLoader

from lib.autoregressive import Autoregressive
from lib.data_processor import DataProcessor
from lib.window_generator import WindowGenerator

with open("config.yaml") as f:
    config = yaml.load(f, Loader=FullLoader)

now = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
model_dir = f"{config['model_directory']}/{config['description']}_{now}/"
dp = DataProcessor(src=config["src_data"], ema_period=config["ema_period"])


def compile_and_fit(model, name: str, train, val, test):
    tb_callback = keras.callbacks.TensorBoard(
        log_dir=f"{config['log_dir']}/{config['description']}_{now}/{name}",
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
        train,
        epochs=config["episodes"],
        validation_data=val,
        callbacks=[tb_callback, early_stopping],
    )
    model.evaluate(
        test,
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


print("\n--------------------------- Close LSTM ---------------------------")
columns = ["day_sin", "day_cos", "high", "low", "close_pct", "close_ema", "volume"]
wg_close = WindowGenerator(
    input_width=config["sequence_length"],
    label_width=config["prediction_length"],
    data_columns=columns,
    label_columns=["close_pct"],
    batch_size=config["batch_size"],
)
lstm_close = single_shot()
compile_and_fit(
    model=lstm_close,
    name="lstm_close",
    train=wg_close.make_dataset(dp.train_df),
    val=wg_close.make_dataset(dp.val_df),
    test=wg_close.make_dataset(dp.test_df),
)

print("\n--------------------------- Open LSTM ---------------------------")
columns = ["day_sin", "day_cos", "high", "low", "open_pct", "open_ema", "volume"]
wg_open = WindowGenerator(
    input_width=config["sequence_length"],
    label_width=config["prediction_length"],
    data_columns=columns,
    batch_size=config["batch_size"],
    shift=config["prediction_length"] + 1,
    label_columns=["open_pct"],
)
lstm_open = single_shot()
compile_and_fit(
    model=lstm_open,
    name="lstm_open",
    train=wg_open.make_dataset(dp.train_df),
    val=wg_open.make_dataset(dp.val_df),
    test=wg_open.make_dataset(dp.test_df),
)

print("\n--------------------------- Autoregressive ---------------------------")
columns = ["open_ema"]
wg_ar = WindowGenerator(
    input_width=config["sequence_length"],
    label_width=config["prediction_length"],
    data_columns=columns,
    batch_size=config["batch_size"],
)
ar_model = Autoregressive(32, config["prediction_length"], len(columns))
compile_and_fit(
    model=ar_model,
    name="autoregressive",
    train=wg_ar.make_dataset(dp.train_df[["close_ema"]]),
    val=wg_ar.make_dataset(dp.val_df[["close_ema"]]),
    test=wg_ar.make_dataset(dp.test_df[["close_ema"]]),
)

print("\n--------------------------- GRU Sentiment ---------------------------")
columns = ["open_ema", "sentiment"]
dp_sentiment = DataProcessor(
    src=config["sentiment_data"], ema_period=config["ema_period"]
)
wg_gru = WindowGenerator(
    input_width=config["sequence_length"],
    label_width=config["prediction_length"],
    data_columns=columns,
    batch_size=config["batch_size"],
    label_columns=["close_ema"],
)
gru_model = keras.Sequential(
    [
        keras.layers.GRU(32, return_sequences=False),
        keras.layers.Dense(
            config["prediction_length"],
            kernel_initializer=keras.initializers.zeros(),
        ),
    ]
)
compile_and_fit(
    model=gru_model,
    name="gru_sentiment",
    train=wg_gru.make_dataset(dp_sentiment.train_df[["close", "sentiment"]]),
    val=wg_gru.make_dataset(dp_sentiment.val_df[["close", "sentiment"]]),
    test=wg_gru.make_dataset(dp_sentiment.test_df[["close", "sentiment"]]),
)
