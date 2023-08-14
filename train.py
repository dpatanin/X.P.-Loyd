import copy
import multiprocessing
from datetime import datetime

import keras
import numpy as np

import lib.replay as replay_lib
from lib.autoregressive import Autoregressive
from lib.data_processor import DataProcessor
from lib.ensemble import Ensemble
from lib.greedy_actor import R2d2EpsilonGreedyActor
from lib.R2D2 import Actor, Learner, R2d2DqnMlpNet, TransitionStructure
from lib.trading_env import TradingEnvironment
from lib.train_loop import run_parallel_training_iterations
from lib.window_generator import WindowGenerator

now = datetime.now().strftime("%d_%m_%Y %H_%M_%S")

DESC = "Post-Ray"
FULL_DATA = "https://onedrive.live.com/download?resid=2ba9b2044e887de1%21290022&authkey=!ADgq6YFliQNylSM"  # No sentiment but ~15 years
SENTIMENT_DATA = "https://onedrive.live.com/download?resid=2ba9b2044e887de1%21293628&authkey=!ANbFvs1RrC9WQ3c"  # With sentiment but ~5 years

EPOCHS = 10
SEQ_LENGTH = 30
PRED_LENGTH = 15
BATCH_SIZE = 512
EMA_PERIOD = 5

# dp_full = DataProcessor("source.csv", EMA_PERIOD)
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
lstm_open_columns = [
    "day_sin",
    "day_cos",
    "high_diff",
    "low_diff",
    "open_pct",
    "open_ema",
    "volume",
]
ar_columns = ["close_ema"]
gru_columns = ["close_ema", "sentiment"]


############################# DNN Stuff #############################


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


def run_forecast():
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
                    label_columns=["close_ema"],
                ),
                dp_sentiment,
            ),
        },
    ]

    for config in configs:
        compile_and_fit(config)


############################# RL Stuff #############################


def env_creator(ensemble: Ensemble):
    return TradingEnvironment(
        df=dp_sentiment.train_df,
        window_size=ensemble.max_window_size,
        forecast_cb=ensemble.forecast,
        forecast_length=PRED_LENGTH,
    )


def run_trading():
    ensemble = Ensemble(
        lstm_model_paths_and_columns=[
            (
                "./models/Post-Ray_lstm_close__13_08_2023 21_01_08.keras",
                lstm_close_columns,
            ),
            (
                "./models/Post-Ray_lstm_open__13_08_2023 21_01_08.keras",
                lstm_open_columns,
            ),
        ],
        ar_model_path="./models/Post-Ray_autoregressive__13_08_2023 21_01_08.keras",
        ar_columns=ar_columns,
        gru_model_path="./models/Post-Ray_gru_sentiment__13_08_2023 21_01_08.keras",
        gru_columns=gru_columns,
        lstm_window=SEQ_LENGTH,
        ar_window=SEQ_LENGTH,
        gru_window=SEQ_LENGTH,
    )

    eval_env = env_creator(ensemble)
    random_state = np.random.RandomState(np.random.seed())

    network = R2d2DqnMlpNet(state_dim=eval_env.state_dim, action_dim=3)
    optimizer = keras.optimizers.Nadam(learning_rate=0.0002)

    replay = replay_lib.PrioritizedReplay(
        capacity=3000,
        structure=TransitionStructure,
        priority_exponent=0.6,
        importance_sampling_exponent=lambda: 5,
        normalize_weights=True,
        random_state=random_state,
        time_major=True,
    )

    # Create queue to shared transitions between actor and learner
    data_queue = multiprocessing.Queue(maxsize=2)

    # Create shared objects so all actor processes can access them
    manager = multiprocessing.Manager()

    # Store copy of latest parameters of the neural network in a shared dictionary, so actors can later access it
    shared_params = manager.dict({"network": None})

    # Create R2D2 learner instance
    learner_agent = Learner(
        network=network,
        optimizer=optimizer,
        replay=replay,
        min_replay_size=1000,
        target_net_update_interval=1000,
        discount=0.99,
        burn_in=32,
        priority_eta=0.9,
        rescale_epsilon=0.0001,
        batch_size=BATCH_SIZE,
        n_step=3,
        clip_grad=True,
        max_grad_norm=1,
        shared_params=shared_params,
    )

    actor_env = env_creator(ensemble)
    actor = Actor(
        data_queue=data_queue,
        network=copy.deepcopy(network),
        random_state=random_state,
        action_dim=3,
        unroll_length=10,
        burn_in=32,
        epsilon=0.4,
        actor_update_interval=100,
        shared_params=shared_params,
    )

    # Create evaluation agent instance
    eval_agent = R2d2EpsilonGreedyActor(
        network=network,
        exploration_epsilon=0.4,
        random_state=random_state,
    )

    run_parallel_training_iterations(
        num_iterations=EPOCHS,
        num_train_steps=EPOCHS,
        num_eval_steps=1,
        learner_agent=learner_agent,
        eval_agent=eval_agent,
        eval_env=eval_env,
        actor=actor,
        actor_env=actor_env,
        data_queue=data_queue,
        tb_log_dir=f"{DESC}__{now}",
    )


# run_forecast()
run_trading()

# import matplotlib.pyplot as plt
# plt.figure(figsize=(16, 6))
# env.render_all()
# plt.show()
