import shutil
from datetime import datetime

import tensorflow as tf
import yaml
from yaml.loader import FullLoader

from lib.autoregressive import Autoregressive
from lib.data_processor import DataProcessor
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

model_dir = f"{config['model_directory']}/{config['description']}_{now}/"
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=config["log_dir"], update_freq=100)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=2, mode="min"
)
optimizer = tf.keras.optimizers.Adamax()
loss = tf.keras.losses.MeanSquaredError()
metrics = [tf.keras.metrics.MeanAbsoluteError()]

# Autoregressive time series forecast
tb_callback.log_dir = (
    f"{config['log_dir']}/{config['description']}_{now}/autoregressive_{now}"
)
ar_model = Autoregressive(64, config["prediction_length"], dp.num_features)
ar_model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=metrics,
)
ar_model.fit(
    wg.make_dataset(dp.train_df),
    epochs=config["episodes"],
    validation_data=wg.make_dataset(dp.val_df),
    callbacks=[tb_callback, early_stopping],
)
ar_model.evaluate(wg.make_dataset(dp.test_df), verbose=0, callbacks=[tb_callback])

ar_model.save(f"{model_dir}auto_regressive/", save_format="tf")
shutil.copy2("config.yaml", f"{model_dir}auto_regressive/")
