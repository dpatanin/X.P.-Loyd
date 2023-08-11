from datetime import datetime

import keras
from ray.rllib.algorithms.r2d2 import r2d2
from ray.tune.registry import register_env

from lib.autoregressive import Autoregressive
from lib.data_processor import DataProcessor
from lib.ensemble import Ensemble
from lib.trading_env import TradingEnvironment
from lib.window_generator import WindowGenerator

SRC_DATA = "https://onedrive.live.com/download?resid=2ba9b2044e887de1%21290022&authkey=!ADgq6YFliQNylSM"
SENTIMENT_DATA = "https://onedrive.live.com/download?resid=2ba9b2044e887de1%21293628&authkey=!ANbFvs1RrC9WQ3c"

DESC = "EMA-Implementation"
EPOCHS = 10
SEQ_LENGTH = 30
PRED_LENGTH = 15
BATCH_SIZE = 512

now = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
dp = DataProcessor(src=SRC_DATA, ema_period=20)


def compile_and_fit(model, name: str, train, val, test):
    tb_callback = keras.callbacks.TensorBoard(
        log_dir=f"logs/{DESC}__{now}/{name}",
        update_freq=100,
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, mode="min"
    )
    optimizer = keras.optimizers.Nadam()
    loss = keras.losses.MeanSquaredError()
    metrics = [keras.metrics.MeanAbsoluteError()]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(
        train,
        epochs=EPOCHS,
        validation_data=val,
        callbacks=[tb_callback, early_stopping],
    )
    model.evaluate(test, verbose=0, callbacks=[tb_callback])

    model.save(f"models/{DESC}__{now}/{name}/", save_format="tf")


def single_shot():
    return keras.Sequential(
        [
            keras.layers.LSTM(64, return_sequences=False),
            keras.layers.Dense(
                PRED_LENGTH, kernel_initializer=keras.initializers.zeros()
            ),
        ]
    )


print("\n--------------------------- Close LSTM ---------------------------")
lstm_close_columns = [
    "day_sin",
    "day_cos",
    "high_diff",
    "low_diff",
    "close_pct",
    "close_ema",
    "volume",
]
wg_close = WindowGenerator(
    input_width=SEQ_LENGTH,
    label_width=PRED_LENGTH,
    data_columns=lstm_close_columns,
    label_columns=["close_pct"],
    batch_size=BATCH_SIZE,
)
lstm_close = single_shot()
compile_and_fit(
    model=lstm_close,
    name="lstm_close",
    train=wg_close.make_dataset(dp.train_df[lstm_close_columns]),
    val=wg_close.make_dataset(dp.val_df[lstm_close_columns]),
    test=wg_close.make_dataset(dp.test_df[lstm_close_columns]),
)

print("\n--------------------------- Open LSTM ---------------------------")
lstm_open_columns = [
    "day_sin",
    "day_cos",
    "high_diff",
    "low_diff",
    "open_pct",
    "open_ema",
    "volume",
]
wg_open = WindowGenerator(
    input_width=SEQ_LENGTH,
    label_width=PRED_LENGTH,
    data_columns=lstm_open_columns,
    batch_size=BATCH_SIZE,
    shift=PRED_LENGTH + 1,
    label_columns=["open_pct"],
)
lstm_open = single_shot()
compile_and_fit(
    model=lstm_open,
    name="lstm_open",
    train=wg_open.make_dataset(dp.train_df[lstm_open_columns]),
    val=wg_open.make_dataset(dp.val_df[lstm_open_columns]),
    test=wg_open.make_dataset(dp.test_df[lstm_open_columns]),
)

print("\n--------------------------- Autoregressive ---------------------------")
ar_columns = ["close_ema"]
wg_ar = WindowGenerator(
    input_width=SEQ_LENGTH,
    label_width=PRED_LENGTH,
    data_columns=ar_columns,
    batch_size=BATCH_SIZE,
)
ar_model = Autoregressive(32, PRED_LENGTH, len(ar_columns))
compile_and_fit(
    model=ar_model,
    name="autoregressive",
    train=wg_ar.make_dataset(dp.train_df[ar_columns]),
    val=wg_ar.make_dataset(dp.val_df[ar_columns]),
    test=wg_ar.make_dataset(dp.test_df[ar_columns]),
)

print("\n--------------------------- GRU Sentiment ---------------------------")
gru_columns = ["close_ema", "sentiment"]
dp_sentiment = DataProcessor(src=SENTIMENT_DATA, ema_period=20)
wg_gru = WindowGenerator(
    input_width=SEQ_LENGTH,
    label_width=PRED_LENGTH,
    data_columns=gru_columns,
    batch_size=BATCH_SIZE,
    label_columns=["close_ema"],
)
gru_model = keras.Sequential(
    [
        keras.layers.GRU(32, return_sequences=False),
        keras.layers.Dense(PRED_LENGTH, kernel_initializer=keras.initializers.zeros()),
    ]
)
compile_and_fit(
    model=gru_model,
    name="gru_sentiment",
    train=wg_gru.make_dataset(dp_sentiment.train_df[gru_columns]),
    val=wg_gru.make_dataset(dp_sentiment.val_df[gru_columns]),
    test=wg_gru.make_dataset(dp_sentiment.test_df[gru_columns]),
)

print("\n--------------------------- R2D2 Trader ---------------------------")

ensemble = Ensemble(
    lstm_models_and_columns=[
        (lstm_close, lstm_close_columns),
        (lstm_open, lstm_open_columns),
    ],
    ar_model=ar_model,
    ar_columns=ar_columns,
    gru_model=gru_model,
    gru_columns=gru_columns,
    window=SEQ_LENGTH,
)


def env_creator(env_config):
    return TradingEnvironment(dp_sentiment.train_df, ensemble)


register_env("trading_env", env_creator)

r2d2.R2D2(env="trading_env")

# config = (
#     R2D2Config()
#     .environment(TradingEnvironment)
#     .framework("tf")
#     .training(
#         model={
#             "fcnet_hiddens": [64],
#             "fcnet_activation": "linear",
#             "use_lstm": True,
#             "lstm_cell_size": 64,
#             "max_seq_len": SEQ_LENGTH,
#         }
#     )
#     .exploration(explore=True, exploration_config={"epsilon_timesteps": 50000})
# )

# algo = config.build()

# for _ in range(5):
#     print(algo.train())  # 3. train it,

# algo.evaluate()  # 4. and evaluate it.
