import tensorflow as tf
import yfinance as yf

from datetime import datetime
from sklearn.model_selection import train_test_split
from pandas_datareader import data as pdr

# Shows which device is used for operations
# tf.debugging.set_log_device_placement(True)

# Load your data
yf.pdr_override()
data = pdr.get_data_yahoo("AAPL", start="2012-01-01", end="2022-12-31")

# Prepare the data for training
X = data[["Open", "High", "Low", "Volume"]]
y = data["Adj Close"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the model architecture
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, input_shape=(X_train.shape[1],), activation="relu"))
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(1))

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

# Train the model
model.fit(X_train, y_train, epochs=1000, batch_size=32)

# Evaluate the model on the test data
test_loss, test_mae = model.evaluate(X_test, y_test)
print("Test loss:", test_loss)
print("Test MAE:", test_mae)

model.save(f"models/model_AAPL_{datetime.now()}.h5")

# Use the model to make predictions on new data
# future_prices = model.predict(new_data)
