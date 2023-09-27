import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Parameters
price = 'Close'
data = 'data/test/00004.csv'
a,b,d = 1,2,1
lag = 10
forecast_steps = 30

# Read CSV data into DataFrame
df = pd.read_csv(data, index_col='Progress')
dataPoints = len(df)

# Explore data
df[price].plot()
plt.show()

# Model identification
sm.graphics.tsa.plot_acf(df[price], lags=lag)
plt.show()

sm.graphics.tsa.plot_pacf(df[price], lags=lag)
plt.show()

# Model estimation (ARMA modeling using ARIMA class)
order = (a, b, d)  # ARMA(1, 0) as an example, with d=0 for no differencing
model = sm.tsa.ARIMA(df[price], order=order)
results = model.fit()

print(results.summary())

# Model diagnostics
sm.graphics.tsa.plot_acf(results.resid, lags=lag)
plt.show()

# Forecasting
forecast = results.get_forecast(steps=forecast_steps)

# Generate date range for forecasted values
last_progress = df.index[-1]
forecast_date_range = np.linspace(last_progress, last_progress + forecast_steps / dataPoints, forecast_steps)

# Plot actual data and forecast
plt.plot(df.index, df[price], label='Actual')
plt.plot(forecast_date_range, forecast.predicted_mean, label='Forecast', linestyle='dashed')
plt.legend()
plt.show()



