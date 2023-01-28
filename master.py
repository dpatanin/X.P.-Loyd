import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import pickle


class FinancialModel:
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.model = LinearRegression()
        self.data = None

    def load_data(self, file_path: str):
        assert os.path.exists(file_path), f"{file_path} does not exist."
        self.data = pd.read_csv(file_path)

        # Check for the presence of specific columns
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        assert set(required_columns).issubset(
            self.data.columns
        ), f"File {file_path} does not contain the required columns: {required_columns}"

        self.data["diff"] = self.data["Close"] - self.data["Close"].shift(1)
        self.data.dropna(inplace=True)
        self.data[
            ["Open", "High", "Low", "Close", "Volume"]
        ] = self.scaler.fit_transform(
            self.data[["Open", "High", "Low", "Close", "Volume"]]
        )

    def create_windowed_dataset(self):
        self.X = []
        self.y = []
        for i in range(self.window_size, len(self.data)):
            self.X.append(
                self.data[i - self.window_size : i][["Open", "High", "Low", "Volume"]]
            )
            self.y.append(self.data["Close"][i])
        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def split_data(self, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        self.X_train = self.X_train.reshape(
            self.X_train.shape[0], self.X_train.shape[1] * self.X_train.shape[2]
        )
        self.X_test = self.X_test.reshape(
            self.X_test.shape[0], self.X_test.shape[1] * self.X_test.shape[2]
        )

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(y_pred, self.y_test)
        r2 = r2_score(y_pred, self.y_test)
        print("Model score: {:.2f}".format(r2))
        print("Mean Squared Error: {:.2f}".format(mse))

    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, filename):
        with open(filename, "rb") as f:
            self.model = pickle.load(f)

    def fit_model(self):
        self.create_windowed_dataset()
        self.split_data()
        self.train_model()
        self.evaluate_model()


model = FinancialModel()
model.load_data("data/ES_futures_sample/ES_continuous_1min_sample.csv")
model.fit_model()
model.save_model("models/financial_model.pkl")
