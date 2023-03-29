import pandas as pd
from tqdm import tqdm
import random
import yaml
from yaml.loader import FullLoader

total_set = 1  # How much of the total data to use
training_set = 0.7  # Cross training-validation
validation_set = 0.15
test_set = 0.15  # Always latest data
timesteps = 1320

with open("config.yaml") as f:
    config = yaml.load(f, Loader=FullLoader)

if training_set + validation_set + test_set != 1:
    raise ValueError(
        "Data set ratio doesn't add up to 1."
        + "If you want to reduce the total size used, set `total_set` accordingly."
    )

# Suppress warning. We want to use copies.
pd.set_option("mode.chained_assignment", None)

print("Reading data source...")
data = pd.read_csv("data/source.csv")
names = ["%05d" % x for x in range(100000)]

print("Indexing...")
start_indices = data[data["DateTime"].str.endswith("18:00:00")].index.to_list()
end_indices = data[data["DateTime"].str.endswith("16:00:00")].index.to_list()

indices = [(s, "s") for s in start_indices]
indices.extend([(e, "e") for e in end_indices])
indices.sort()

# Remove incomplete indices
anomalies = []
marker = ""
for i in indices:
    if marker == "s" and i[1] == "s":
        anomalies.append(indices[indices.index(i) - 1][0])
    elif marker == "e" and i[1] == "e":
        anomalies.append(i[0])
    marker = i[1]

indices = [i[0] for i in indices if i[0] not in anomalies]
indices = [
    (s, e) for s, e in zip(indices[::2], indices[1::2]) if e - s <= timesteps * 1.0625
]

print("Splitting & shuffling...")
# Use always latest data for test
test_size = round(len(indices) * total_set * test_set)
test_indices = indices[-test_size:-1]

# Shuffle & split remaining data
train_size = round(len(indices) * total_set * training_set)
valid_size = round(len(indices) * total_set * validation_set)
train_valid_indices = indices[:-test_size]
random.shuffle(train_valid_indices)
train_indices = train_valid_indices[:train_size]
valid_indices = train_valid_indices[train_size : valid_size + train_size]


def generate(indices: list[tuple], names: list[str], path: str, label: str):
    for index, name in tqdm(zip(indices, names), desc=label, leave=True):
        start, end = index
        session = data.iloc[start : end + 1]

        # ------ Manipulate data ------
        session.drop(["DateTime"], axis=1, inplace=True)
        progress = [(100 / timesteps) * x for x in range(1, timesteps + 1)]
        session.insert(0, "Progress", progress)

        # Normalizing
        session["Open"] -= session["Open"].iloc[0]
        session["High"] -= session["High"].iloc[0]
        session["Low"] -= session["Low"].iloc[0]
        # Keeping original close prizes for calculations
        session["CloseNorm"] = session["Close"] - session["Close"].iloc[0]
        # -----------------------------

        session.to_csv(f"{path}/{name}.csv", sep=",", index=False)


generate(
    train_indices,
    names[:train_size],
    config["training_data"],
    "Generating training set",
)
generate(
    valid_indices,
    names[train_size : valid_size + train_size],
    config["validation_data"],
    "Generating validation set",
)
generate(
    test_indices,
    names[valid_size + train_size : valid_size + train_size + test_size],
    config["test_data"],
    "Generating test set",
)
