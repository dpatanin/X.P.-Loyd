import pandas as pd
from tqdm import tqdm
import random
import yaml
from yaml.loader import FullLoader

with open("config.yaml") as f:
    config = yaml.load(f, Loader=FullLoader)

if config["training_set"] + config["validation_set"] + config["test_set"] != 1:
    raise ValueError(
        "Data set ration doesn't add up to 1."
        + "If you want to reduce the total size used, set `total_set` accordingly."
    )

# Suppress warning. We want to use copies.
pd.set_option("mode.chained_assignment", None)

print("Reading data source...")
data = pd.read_csv(config["source_data"])
names = ["%05d" % x for x in range(100000)]
start_indices = []
end_indices = []

for idx, row in tqdm(data.iterrows(), desc="Indexing", leave=True):
    if row[config["marker_column"]].endswith(config["session_begin_marker"]):
        start_indices.append(idx)
    if row[config["marker_column"]].endswith(config["session_end_marker"]):
        end_indices.append(idx)

if end_indices[0] < start_indices[0]:
    end_indices.pop(0)
if start_indices[-1] > end_indices[-1]:
    start_indices.pop(-1)
indices = [*zip(start_indices, end_indices)]

print("Splitting & shuffling...")
# Use always latest data for test
test_size = round(len(indices) * config["total_set"] * config["test_set"])
test_indices = indices[-test_size:-1]

# Shuffle & split remaining data
train_size = round(len(indices) * config["total_set"] * config["training_set"])
valid_size = round(len(indices) * config["total_set"] * config["validation_set"])
train_valid_indices = indices[:-test_size]
random.shuffle(train_valid_indices)
train_indices = train_valid_indices[: train_size - 1]
valid_indices = train_valid_indices[train_size : valid_size - 1]


def generate(indices: list[tuple], names: list[str], path: str, label: str):
    for index, name in tqdm(zip(indices, names), desc=label, leave=True):
        start, end = index
        session = data.iloc[start : end + 1]

        # ------ Manipulate data ------
        timesteps = end + 1 - start
        progress = [(100 / timesteps) * x for x in range(1, timesteps + 1)]
        session.drop(["DateTime"], axis=1, inplace=True)
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
    names[: train_size - 1],
    config["training_data"],
    "Generating training set",
)
generate(
    valid_indices,
    names[train_size : valid_size - 1],
    config["validation_data"],
    "Generating validation set",
)
generate(
    test_indices,
    names[valid_size : test_size - 1],
    config["test_data"],
    "Generating test set",
)
