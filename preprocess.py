import pandas as pd
from tqdm import tqdm
import random
import yaml
from yaml.loader import FullLoader
from datetime import datetime, timedelta

TIMESTEPS = 1321
with open("config.yaml") as f:
    config = yaml.load(f, Loader=FullLoader)

print("Reading data source...")
data = pd.read_csv("data/source.csv")
names = ["%05d" % x for x in range(100000)]

print("Indexing...")
start_indices = data[data["dateTime"].str.endswith("18:00:00")].index.to_list()
end_indices = data[data["dateTime"].str.endswith("16:00:00")].index.to_list()

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
    (s, e)
    for s, e in zip(indices[::2], indices[1::2])
    if TIMESTEPS * 0.9375 <= e - s <= TIMESTEPS * 1.0625
]

print("Shuffling...")
random.shuffle(indices)


def interpolate(session: pd.DataFrame):  # sourcery skip: pandas-avoid-inplace
    session_size = len(session.index)

    while session_size != TIMESTEPS:
        interpolations = []
        removals = []
        for i in range(session_size - 2):
            first = session.iloc[i]
            second = session.iloc[i + 1]
            diff = (
                datetime.fromisoformat(second["dateTime"])
                - datetime.fromisoformat(first["dateTime"])
            ).total_seconds() / 60

            if diff > 1:
                inter = first.copy()
                inter["dateTime"] = str(
                    datetime.fromisoformat(inter["dateTime"]) + timedelta(minutes=1)
                )
                inter["volume"] = round((first["volume"] + second["volume"]) / 2)

                for c in ["open", "high", "low", "close"]:
                    inter[c] = round(((first[c] + second[c]) * 4) / 2) / 4

                interpolations.append((session.index[i] + 0.5, inter.tolist()))
            elif diff == 0:
                removals.append(session.index[i])
            elif diff < 0:
                raise RuntimeError("Something went horribly wrong... again...")

        for inter in interpolations:
            session.loc[inter[0]] = inter[1]
        session.drop(removals, axis=0, inplace=True)
        session.sort_index(inplace=True)
        session.reset_index(inplace=True, drop=True)
        session_size = len(session.index)


for index, name in tqdm(zip(indices, names[: len(indices)]), leave=True):
    start, end = index
    session = data.iloc[start : end + 1].copy()

    # ------ Manipulate data ------
    # Interpolate missing data
    if len(session.index) < TIMESTEPS:
        interpolate(session)
    session.drop(["dateTime"], axis=1, inplace=True)

    # Add progress indicator
    progress = [(100 / TIMESTEPS) * x for x in range(1, TIMESTEPS + 1)]
    session.insert(0, "progress", progress)

    # Normalizing
    session["open"] -= session["open"].iloc[0]
    session["high"] -= session["high"].iloc[0]
    session["low"] -= session["low"].iloc[0]
    session["close"] -= session["close"].iloc[0]
    # -----------------------------

    session.to_csv(f"{config['data_dir']}/{name}.csv", sep=",", index=False)
