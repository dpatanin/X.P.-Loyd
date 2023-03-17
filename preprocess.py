import pandas as pd
from tqdm import tqdm

pd.set_option("mode.chained_assignment", None)

# Simple script for splitting one datafile into trading sessions

data = pd.read_csv("")
names = ["%05d" % x for x in range(100000)]
start_indices = []
end_indices = []

for idx, row in tqdm(data.iterrows(), desc="Indexing", leave=True):
    if row["DateTime"].endswith("18:00:00"):
        start_indices.append(idx)
    if row["DateTime"].endswith("16:59:00"):
        end_indices.append(idx)

if end_indices[0] < start_indices[0]:
    end_indices.pop(0)
if start_indices[-1] > end_indices[-1]:
    start_indices.pop(-1)

names = names[: len(start_indices)]
for start, end, name in tqdm(
    zip(start_indices, end_indices, names), desc="Splitting", leave=True
):
    timesteps = end + 1 - start
    progress = [(100 / timesteps) * x for x in range(1, timesteps + 1)]

    session = data.iloc[start : end + 1]
    session.drop(["DateTime"], axis=1, inplace=True)
    session.insert(0, "Progress", progress)
    session.to_csv(f"data/{name}.csv", sep=",", index=False)
