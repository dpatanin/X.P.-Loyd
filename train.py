from datetime import datetime

import keras
import ray
import tensorflow as tf
from ray import air
from ray.air.config import ScalingConfig
from ray.air.integrations.keras import ReportCheckpointCallback
from ray.rllib.algorithms.r2d2 import R2D2Config
from ray.train.tensorflow import TensorflowTrainer
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.tune.tuner import TuneConfig, Tuner

from lib.autoregressive import Autoregressive
from lib.data_processor import DataProcessor
from lib.ensemble import EnsembleConfig
from lib.trading_env import TradingEnvironment
from lib.window_generator import WindowGenerator

ray.init(num_gpus=1)
now = datetime.now().strftime("%d_%m_%Y %H_%M_%S")

DESC = "Ray-Integration"
FULL_DATA = "https://onedrive.live.com/download?resid=2ba9b2044e887de1%21290022&authkey=!ADgq6YFliQNylSM"  # No sentiment but ~15 years
SENTIMENT_DATA = "https://onedrive.live.com/download?resid=2ba9b2044e887de1%21293628&authkey=!ANbFvs1RrC9WQ3c"  # With sentiment but ~5 years

EPOCHS = 1
SEQ_LENGTH = 30
PRED_LENGTH = 15
BATCH_SIZE = 512
EMA_PERIOD = 5

dp_full = DataProcessor("source.csv", EMA_PERIOD)
dp_sentiment = DataProcessor("source_sentiment_merge.csv", EMA_PERIOD)

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

ar_columns = ["close_ema"]
wg_ar = WindowGenerator(
    input_width=SEQ_LENGTH,
    label_width=PRED_LENGTH,
    data_columns=ar_columns,
    batch_size=BATCH_SIZE,
)

gru_columns = ["close_ema", "sentiment"]
wg_gru = WindowGenerator(
    input_width=SEQ_LENGTH,
    label_width=PRED_LENGTH,
    data_columns=gru_columns,
    batch_size=BATCH_SIZE,
    label_columns=["close_ema"],
)


def get_dataset_maker(wg: WindowGenerator, dp: DataProcessor):
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


def ar():
    return Autoregressive(32, PRED_LENGTH, len(ar_columns))


def train_func(config: dict):
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        multi_worker_model = config["model"]()
        multi_worker_model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=config["lr"]),
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.MeanSquaredError(name="loss")],
        )

    for _ in range(EPOCHS):
        multi_worker_model.fit(
            config["datasets"]["ds_train"](),
            callbacks=[ReportCheckpointCallback()],
            verbose=0,
            validation_data=config["datasets"]["ds_val"](),
        )


lstm_close_config = {
    "name": "lstm_close",
    "model": single_shot,
    "lr": 0.0002,
    "datasets": get_dataset_maker(wg_close, dp_full),
}
lstm_open_config = {
    "name": "lstm_open",
    "model": single_shot,
    "lr": 0.0002,
    "datasets": get_dataset_maker(wg_open, dp_full),
}
ar_config = {
    "name": "autoregressive",
    "model": ar,
    "lr": 0.0002,
    "datasets": get_dataset_maker(wg_ar, dp_full),
}
gru_config = {
    "name": "gru_sentiment",
    "model": gru,
    "lr": 0.0002,
    "datasets": get_dataset_maker(wg_gru, dp_sentiment),
}

scaling_config = ScalingConfig(
    num_workers=1,
    use_gpu=True,
    resources_per_worker={"CPU": 8, "GPU": 1},
)

param_space = {
    "train_loop_config": {"lr": 0.0002},
    "scaling_config": scaling_config,
}

for config in [lstm_close_config, lstm_open_config, ar_config, gru_config]:
    trainer = TensorflowTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=config,
        scaling_config=scaling_config,
    )

    tuner = Tuner(
        trainer,
        param_space=param_space,
        tune_config=TuneConfig(num_samples=1, metric="loss", mode="min"),
        run_config=air.RunConfig(name=f"{DESC}_{config['name']}__{now}"),
    )
    result_grid = tuner.fit()
    best_result = result_grid.get_best_result()
    print("Best Result:", best_result)


ensemble_config: EnsembleConfig = {
    "lstm_model_paths_and_columns": [
        (
            "./models/Ray-Integration_lstm_close__13_08_2023 16_00_10/TensorflowTrainer_c5666_00000_0_2023-08-13_16-00-33/checkpoint_000000/dict_checkpoint.pkl",
            lstm_close_columns,
        ),
        (
            "./models/Ray-Integration_lstm_open__13_08_2023 16_00_10/TensorflowTrainer_59ce7_00000_0_2023-08-13_16-04-42/checkpoint_000000/dict_checkpoint.pkl",
            lstm_open_columns,
        ),
    ],
    "ar_model_path": "./models/Ray-Integration_autoregressive__13_08_2023 16_00_10/TensorflowTrainer_f81c2_00000_0_2023-08-13_16-09-08/checkpoint_000000/dict_checkpoint.pkl",
    "ar_columns": ar_columns,
    "gru_model_path": "./models/Ray-Integration_gru_sentiment__13_08_2023 16_00_10/TensorflowTrainer_cf104_00000_0_2023-08-13_16-29-27/checkpoint_000000/dict_checkpoint.pkl",
    "gru_columns": gru_columns,
    "lstm_window": SEQ_LENGTH,
    "ar_window": SEQ_LENGTH,
    "gru_window": SEQ_LENGTH,
}


def env_creator(env_config):
    return TradingEnvironment(dp_sentiment.train_df, ensemble_config, PRED_LENGTH)


register_env("trading_env", env_creator)

algo = (
    R2D2Config()
    .environment("trading_env")
    .framework("tf")
    .training(
        model={
            "fcnet_hiddens": [64],
            "fcnet_activation": "linear",
            "use_lstm": True,
            "lstm_cell_size": 64,
        }
    )
    .resources(num_gpus=1, num_learner_workers=0)
    .build()
)

for _ in range(EPOCHS):
    result = algo.train()
    print(pretty_print(result))

    checkpoint_dir = algo.save(f"/models/{DESC}__{now}")
    print(f"Checkpoint saved in directory {checkpoint_dir}")


# import matplotlib.pyplot as plt
# plt.figure(figsize=(16, 6))
# env.render_all()
# plt.show()
