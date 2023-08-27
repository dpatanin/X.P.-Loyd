import pandas as pd

class SMACrossOver:
    def __init__(self, data_frame):
        self.data_frame = data_frame
        self.data_frame['position'] = None

    def SMA(self, data, period):
        return data.rolling(window=period).mean()

    def EnterPosition(self, row):
        #Enter Long
        if row['SMADiff'] > 1:
            return 1
        #Enter Short
        elif row['SMADiff'] < 1:
            return 2
        return None


    def analyze(self, periodFast, periodSlow):
        self.data_frame['SMADiff'] = self.SMA(self.data_frame['close'], periodFast) / self.SMA(self.data_frame['close'], periodSlow)
        self.data_frame['position'] = self.data_frame.apply(self.EnterPosition, axis= 1)


if __name__ == '__main__':

    # Example usage
    data = {'open': [105, 110, 100, 98, 105, 107, 103, 112, 118, 115],
            'high': [108, 115, 102, 104, 110, 112, 105, 118, 122, 118],
            'low': [102, 108, 97, 95, 100, 104, 99, 110, 112, 110],
            'close': [103, 112, 99, 100, 106, 108, 101, 115, 120, 114]}

    df = pd.DataFrame(data)

    strategy = SMACrossOver(df)
    strategy.analyze(periodFast=3, periodSlow=5)
    print(df)