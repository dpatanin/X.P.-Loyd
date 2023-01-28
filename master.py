import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


class FinancialModel:
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.model = LinearRegression()

    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)
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
            self.X, self.y, test_size=test_size
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
        score = self.model.score(self.X_test, self.y_test)
        print("Model score: {:.2f}".format(score))


# Usage example
model = FinancialModel()
model.load_data("futures.csv")
model.create_windowed_dataset()
model.split_data()
model.train_model()
model.evaluate_model()
