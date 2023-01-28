import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load the data
data = pd.read_csv("futures.csv")

# create a new column with the difference of the close price
data['diff'] = data['Close'] - data['Close'].shift(1)

# drop missing values
data.dropna(inplace=True)

# normalize the data
scaler = MinMaxScaler()
data[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])

# create the windowed dataset
window_size = 60
X = []
y = []
for i in range(window_size, len(data)):
    X.append(data[i-window_size:i][['Open', 'High', 'Low', 'Volume']])
    y.append(data['Close'][i])

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.2)

# reshape dim-3 array to dim-2
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])

# create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# evaluate the model
score = model.score(X_test, y_test)
print("Model score: {:.2f}".format(score))
