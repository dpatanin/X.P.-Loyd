from datetime import datetime

import tensorflow as tf
import yaml
from yaml.loader import FullLoader

from lib.data_processor import DataProcessor
from lib.trader import PredictionModel
from lib.window_generator import WindowGenerator

with open("config.yaml") as f:
    config = yaml.load(f, Loader=FullLoader)

now = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
dp = DataProcessor(config["data_src"])
wg = WindowGenerator(
    input_width=config["sequence_length"],
    label_width=config["prediction_length"],
    data_columns=dp.train_df.columns,
    label_columns=["close"],
    batch_size=config["batch_size"],
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=config["log_dir"])
trader = PredictionModel(
    sequence_length=config["sequence_length"],
    num_features=dp.num_features,
    learning_rate=config["agent"]["learning_rate"],
    num_output=config["prediction_length"],
)
trader.summary()

trader.fit(
    wg.make_dataset(dp.train_df),
    epochs=10,
    validation_data=wg.make_dataset(dp.val_df),
    callbacks=[tensorboard_callback],
)

trader.evaluate(wg.make_dataset(dp.val_df), callbacks=[tensorboard_callback])
trader.evaluate(
    wg.make_dataset(dp.test_df), verbose=0, callbacks=[tensorboard_callback]
)
