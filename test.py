from src.data_processor import DataProcessor

dp = DataProcessor('data', ["Open", "High", "Low", "Close", "Volume"], 5, 2)

batch = dp.load_batch(1)

print(batch[0])