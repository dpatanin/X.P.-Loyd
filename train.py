import math

from lib.data_processor import DataProcessor
from lib.trader import FreeLaborTrader
from lib.state import State
from lib.action_space import ActionSpace
from lib.progress_bar import ProgressBar
import tensorflow as tf
import pandas as pd
import numpy as np
import yaml
from yaml.loader import FullLoader
from datetime import datetime
import time
import os

with open("config.yaml") as f:
    config = yaml.load(f, Loader=FullLoader)


def num_features() -> int:
    return len(State(data=empty_sequence(), tick_size=0, tick_value=0).to_df().columns)


def empty_sequence() -> pd.DataFrame:
    empty = np.zeros((config["sequence_length"], len(config["data_headers"])))
    return pd.DataFrame(empty, columns=config["data_headers"])


def calc_terminal_reward(reward: float, state: "State") -> float:
    if state.contracts != 0:
        reward += state.exit_position()
    if state.balance < 0:
        return -10000000000000000000000000
    else:
        return (reward + state.balance - config["initial_balance"]) * config[
            "reward_factors"
        ]["session_total"]


def rem_time(times: list[int], it_left: int):
    rem_time_sec = it_left * (sum(times) / len(times))
    return f"Remaining time: {math.floor(rem_time_sec / 3600)} h {math.floor(rem_time_sec / 60) % 60} min"


def saved_model():
    versions = []
    versions.extend(int(item) for item in os.listdir("./models/") if (item.isdigit()))

    tf.saved_model.save(
        trader.model, f'./{config["model_directory"]}/{max(versions) + 1}'
    )


def init_states() -> list[State]:
    return [
        State(
            data=empty_sequence(),
            balance=config["initial_balance"],
            tick_size=config["tick_size"],
            tick_value=config["tick_value"],
        )
    ] * config["batch_size"]


action_space = ActionSpace(
    threshold=config["action_space"]["threshold"],
    limit=config["action_space"]["trade_limit"],
    intrinsic_fac=config["reward_factors"]["intrinsic"],
)
dp = DataProcessor(
    headers=config["data_headers"],
    sequence_length=config["sequence_length"],
    batch_size=config["batch_size"],
    dir=config["data_dir"],
)
trader = FreeLaborTrader(
    sequence_length=config["sequence_length"],
    batch_size=config["batch_size"],
    num_features=num_features(),
    memory_size=config["agent"]["memory_size"],
    update_freq=config["agent"]["update_frequency"],
    hindsight_reward_fac=config["reward_factors"]["hindsight"],
    gamma=config["agent"]["gamma"],
    epsilon=config["agent"]["epsilon"],
    epsilon_final=config["agent"]["epsilon_final"],
    epsilon_decay=config["agent"]["epsilon_decay"],
    learning_rate=config["agent"]["learning_rate"],
)
now = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
trader.model.summary()

########################### Training ###########################

# trader.load("models/[name].h5")

dp.batched_dir = dp.batch_dir()
pbar = ProgressBar(
    episodes=config["episodes"],
    batches=len(dp.batched_dir),
    sequences_per_batch=len(dp.load_batch(0)),
    prefix="Training",
    suffix="Remaining time: ???",
    leave=True,
)

rem_batches = config["episodes"] * len(dp.batched_dir)
times_per_batch = []

for e in range(1, config["episodes"] + 1):
    for i in range(len(dp.batched_dir)):
        t = time.time()
        batch = dp.load_batch(i)
        states = init_states()

        for idx, sequences in enumerate(batch):
            for seq, state in zip(sequences, states):
                state.data = seq
            snapshot = states.copy()  # States before action; For experiences

            q_values = trader.predict(states)
            rewards = [
                action_space.take_action(q, s)[0] for q, s in zip(q_values, states)
            ]

            done = idx == len(batch) - 1
            if done:
                rewards = [calc_terminal_reward(r, s) for r, s in zip(rewards, states)]

            for snap, reward, state in zip(snapshot, rewards, states.copy()):
                trader.memory.add((snap, reward, state, done))

            if len(trader.memory) > config["batch_size"]:
                trader.batch_train()

            pbar.update(e, i + 1, idx + 1)

        # Create hindsight experiences
        trader.memory.analyze_missed_opportunities(action_space)

        rem_batches -= 1
        times_per_batch.append((time.time() - t))
        pbar.suffix = rem_time(times_per_batch, rem_batches)

    if e < config["episodes"]:
        trader.model.save(
            f"{config['model_directory']}/{config['model_name']}_ep{e}_{now}.h5"
        )
    else:
        trader.model.save(
            f"{config['model_directory']}/{config['model_name']}_terminal_{now}.h5"
        )
        saved_model()  # Save the model for tensorflow-serving

pbar.close()


########################### Validation ###########################

pbar = ProgressBar(
    episodes=1,
    batches=len(dp.batched_dir),
    sequences_per_batch=len(dp.load_batch(0)),
    prefix="Validation",
    suffix="Remaining time: ???",
    leave=True,
)

rem_batches = len(dp.batched_dir)
trader.memory.clear()
trader.epsilon = 0  # This removes random choices
balance_list = pd.DataFrame()
action_list = pd.DataFrame()
times_per_batch = []


for i in range(len(dp.batched_dir)):
    t = time.time()
    batch = dp.load_batch(i)
    states = init_states()

    for idb, s in enumerate(states):
        # +1 to keep initial balance
        balance_list[f"b{i}s{idb}"] = [s.balance] * (len(batch) + 1)
        action_list[f"b{i}s{idb}"] = ["STAY"] * (len(batch) + 1)

    for idx, sequences in enumerate(batch):
        for seq, state in zip(sequences, states):
            state.data = seq

        q_values = trader.predict(states)
        for ids, qs in enumerate(zip(q_values, states)):
            amount = action_space.calc_trade_amount(qs[0], qs[1])
            action = action_space.take_action(qs[0], qs[1])[1]
            action_list[f"b{i}s{ids}"].iloc[idx + 1] = f"{action}|{amount}"
            balance_list[f"b{i}s{ids}"].iloc[idx + 1] = qs[1].balance

        pbar.update(batch=i + 1, seq=idx + 1)

    rem_batches -= 1
    times_per_batch.append((time.time() - t))
    pbar.suffix = rem_time(times_per_batch, rem_batches)

pbar.close()
writer = pd.ExcelWriter(
    f"data/validation_{config['model_name']}_{now}.xlsx", engine="xlsxwriter"
)
action_list.to_excel(writer, sheet_name="actions", index=False)
balance_list.to_excel(writer, sheet_name="balances", index=False)
writer.close()
