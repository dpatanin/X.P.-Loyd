from datetime import datetime

import keras

from lib.autoregressive import Autoregressive
from lib.data_processor import DataProcessor
from lib.window_generator import WindowGenerator

now = datetime.now().strftime("%d_%m_%Y %H_%M_%S")

DESC = "Post-Ray"
FULL_DATA = "https://onedrive.live.com/download?resid=2ba9b2044e887de1%21290022&authkey=!ADgq6YFliQNylSM"  # No sentiment but ~15 years
SENTIMENT_DATA = "https://onedrive.live.com/download?resid=2ba9b2044e887de1%21293628&authkey=!ANbFvs1RrC9WQ3c"  # With sentiment but ~5 years

EPOCHS = 10
SEQ_LENGTH = 30
PRED_LENGTH = 15
BATCH_SIZE = 512

dp_full = DataProcessor(FULL_DATA)
dp_sentiment = DataProcessor(SENTIMENT_DATA)


def dataset_maker(wg: WindowGenerator, dp: DataProcessor):
    return {
        "ds_train": lambda: wg.make_dataset(dp.train_df),
        "ds_val": lambda: wg.make_dataset(dp.val_df),
        "ds_test": lambda: wg.make_dataset(dp.test_df),
    }


def single_shot():
    return keras.Sequential(
        [
            keras.layers.LSTM(64, return_sequences=False),
            keras.layers.Dense(
                PRED_LENGTH, kernel_initializer=keras.initializers.zeros()
            ),
        ]
    )


def gru():
    return keras.Sequential(
        [
            keras.layers.GRU(32, return_sequences=False),
            keras.layers.Dense(
                PRED_LENGTH, kernel_initializer=keras.initializers.zeros()
            ),
        ]
    )


def compile_and_fit(config: dict):
    tb_callback = keras.callbacks.TensorBoard(
        log_dir=f"logs/{DESC}__{now}/{config['name']}",
        update_freq=100,
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, mode="min"
    )
    optimizer = keras.optimizers.Nadam(learning_rate=0.0002)
    loss = keras.losses.MeanSquaredError()
    metrics = [keras.metrics.MeanSquaredError(name="loss")]

    model = config["model"]()
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(
        config["datasets"]["ds_train"](),
        epochs=EPOCHS,
        validation_data=config["datasets"]["ds_val"](),
        callbacks=[tb_callback, early_stopping],
    )
    model.evaluate(config["datasets"]["ds_test"](), verbose=0, callbacks=[tb_callback])

    model.save(f"models/{DESC}_{config['name']}__{now}.keras")


lstm_close_columns = [
    "day_sin",
    "day_cos",
    "high_diff",
    "low_diff",
    "close_pct",
    "volume",
]
lstm_open_columns = [
    "day_sin",
    "day_cos",
    "high_diff",
    "low_diff",
    "open_pct",
    "volume",
]
ar_columns = ["close_pct"]
gru_columns = ["close_pct", "sentiment"]

configs = [
    {
        "name": "lstm_close",
        "model": single_shot,
        "datasets": dataset_maker(
            WindowGenerator(
                input_width=SEQ_LENGTH,
                label_width=PRED_LENGTH,
                data_columns=lstm_close_columns,
                label_columns=["close_pct"],
                batch_size=BATCH_SIZE,
            ),
            dp_full,
        ),
    },
    {
        "name": "lstm_open",
        "model": single_shot,
        "datasets": dataset_maker(
            WindowGenerator(
                input_width=SEQ_LENGTH,
                label_width=PRED_LENGTH,
                data_columns=lstm_open_columns,
                batch_size=BATCH_SIZE,
                shift=PRED_LENGTH + 1,
                label_columns=["open_pct"],
            ),
            dp_full,
        ),
    },
    {
        "name": "autoregressive",
        "model": lambda: Autoregressive(32, PRED_LENGTH, len(ar_columns)),
        "datasets": dataset_maker(
            WindowGenerator(
                input_width=SEQ_LENGTH,
                label_width=PRED_LENGTH,
                data_columns=ar_columns,
                batch_size=BATCH_SIZE,
            ),
            dp_full,
        ),
    },
    {
        "name": "gru_sentiment",
        "model": gru,
        "datasets": dataset_maker(
            WindowGenerator(
                input_width=SEQ_LENGTH,
                label_width=PRED_LENGTH,
                data_columns=gru_columns,
                batch_size=BATCH_SIZE,
                label_columns=["close_pct"],
            ),
            dp_sentiment,
        ),
    },
]

for config in configs:
    compile_and_fit(config)
