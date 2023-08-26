import pandas as pd

class SMACrossOver:
    def __init__(self, data_frame):
        self.data_frame = data_frame
        self.data_frame['position'] = None
        self.short_condition = False
        self.long_condition = True

    def SMA(self, data, period):
        return data.rolling(window=period).mean()

    def CrossAbove(self, x, y):
        return x > y

    def CrossBelow(self, x, y):
        return x < y

    def EnterShort(self):
        self.data_frame['position'] = self.data_frame['position'].mask(self.short_condition, 2)

    def EnterLong(self):
        self.data_frame['position'] = self.data_frame['position'].mask(self.long_condition, 1)


    def analyze(self, periodFast, periodSlow):
        self.data_frame['smaFast'] = self.SMA(self.data_frame['close'], periodFast)
        self.data_frame['smaSlow'] = self.SMA(self.data_frame['close'], periodSlow)

        self.long_condition = self.CrossAbove(self.data_frame['smaFast'], self.data_frame['smaSlow'])
        self.short_condition = self.CrossBelow(self.data_frame['smaFast'], self.data_frame['smaSlow'])

        self.EnterLong()
        self.EnterShort()

# Example usage
data = {'open': [105, 110, 100, 98, 105, 107, 103, 112, 118, 115],
        'high': [108, 115, 102, 104, 110, 112, 105, 118, 122, 118],
        'low': [102, 108, 97, 95, 100, 104, 99, 110, 112, 110],
        'close': [103, 112, 99, 100, 106, 108, 101, 115, 120, 114]}

df = pd.DataFrame(data)

strategy = SMACrossOver(df)
strategy.analyze(periodFast=3, periodSlow=5)
print(df)