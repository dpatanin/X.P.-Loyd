import pandas as pd

class SMACrossOver:
    def __init__(self, data_frame):
        self.data_frame = data_frame
        self.long_triggered = False
        self.short_triggered = False

    def SMA(self, data, period):
        return None if len(data) < period else sum(data[-period:]) / period

    def CrossAbove(self, x, y):
        if not self.long_triggered and x > y:
            self.long_triggered = True
            self.short_triggered = False
            return True
        return False

    def CrossBelow(self, x, y):
        if not self.short_triggered and x < y:
            self.short_triggered = True
            self.long_triggered = False
            return True
        return False

    def EnterShort(self, index):
        self.data_frame.at[index, 'position'] = -1

    def EnterLong(self, index):
        self.data_frame.at[index, 'position'] = 1


    def analyze(self, periodFast, periodSlow):
        self.data_frame['smaFast'] = self.data_frame['close'].rolling(window=periodFast).mean()
        self.data_frame['smaSlow'] = self.data_frame['close'].rolling(window=periodSlow).mean()

        for index, row in self.data_frame.iterrows():
            if self.CrossAbove(row['smaFast'], row['smaSlow']):
                self.EnterLong(index)
            elif self.CrossBelow(row['smaFast'], row['smaSlow']):
                self.EnterShort(index)

# Example usage
data = {'open': [105, 110, 100, 98, 105, 107, 103, 112, 118, 115],
        'high': [108, 115, 102, 104, 110, 112, 105, 118, 122, 118],
        'low': [102, 108, 97, 95, 100, 104, 99, 110, 112, 110],
        'close': [103, 112, 99, 100, 106, 108, 101, 115, 120, 114]}

df = pd.DataFrame(data)

strategy = SMACrossOver(df)
strategy.analyze(periodFast=3, periodSlow=5)
print(df)