import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from yaml.loader import FullLoader

from lib.action_space import ActionSpace
from lib.data_processor import DataProcessor
from lib.experience_replay import Memory
from lib.progress_bar import ProgressBar
from lib.state import State
from lib.trader import FreeLaborTrader

with open("config.yaml") as f:
    config = yaml.load(f, Loader=FullLoader)


def num_features() -> int:
    return len(State(data=empty_sequence(), tick_size=0, tick_value=0).to_df().columns)


def empty_sequence() -> pd.DataFrame:
    empty = np.zeros((config["sequence_length"], len(config["data_headers"])))
    return pd.DataFrame(empty, columns=config["data_headers"])


def rem_time(times: list[int], it_left: int):
    rem_time_sec = it_left * (sum(times) / len(times))
    return f"Remaining time: {math.floor(rem_time_sec / 3600)} h {math.floor(rem_time_sec / 60) % 60} min"


def saved_model():
    versions = []
    versions.extend(int(item) for item in os.listdir("./models/") if (item.isdigit()))

    tf.saved_model.save(
        trader.model,
        f'./{config["model_directory"]}/{max(versions) + 1 if versions else 1}',
    )


def init_states(amount: int) -> list[State]:
    return [
        State(
            data=empty_sequence(),
            balance=config["initial_balance"],
            tick_size=config["tick_size"],
            tick_value=config["tick_value"],
        )
        for _ in range(amount)
    ]


action_space = ActionSpace(
    threshold=config["action_space"]["threshold"],
    limit=config["action_space"]["trade_limit"],
)

dp = DataProcessor(
    headers=config["data_headers"],
    sequence_length=config["sequence_length"],
    batch_size=config["batch_size"],
    dir=config["data_training"],
)
sequences_per_batch = len(dp.load_batch(0))

trader = FreeLaborTrader(
    sequence_length=config["sequence_length"],
    batch_size=config["batch_size"],
    num_features=num_features(),
    update_freq=config["agent"]["update_frequency"],
    gamma=config["agent"]["gamma"],
    epsilon=config["agent"]["epsilon"],
    epsilon_final=config["agent"]["epsilon_final"],
    epsilon_decay=config["agent"]["epsilon_decay"],
    learning_rate=config["agent"]["learning_rate"],
)
trader.model.summary()

now = datetime.now().strftime("%d_%m_%Y %H_%M_%S")

########################### Training ###########################

# trader.load("models/[name].h5")

pbar = ProgressBar(
    episodes=config["episodes"],
    batches=len(dp.batched_dir),
    sequences_per_batch=sequences_per_batch,
    prefix="Training",
    leave=True,
)

for e in range(1, config["episodes"] + 1):
    for i in range(len(dp.batched_dir)):
        batch = dp.load_batch(i)
        states = init_states(len(batch[0]))
        memories = [Memory() for _ in states]

        for idx, sequences in enumerate(batch):
            done = idx == len(batch) - 1
            for seq, state, memory in zip(sequences, states, memories):
                state.data = seq
                memory.outcome = state.copy()

            q_values = trader.predict(states)
            for q, s, m in zip(q_values, states, memories):
                m.reward = action_space.take_action(q, s)[0]
                m.done = done

                if m.is_complete():  # This is mainly a check for first iteration
                    trader.memory.add(m.copy())
                m.origin = m.outcome
                m.outcome = None

            if len(trader.memory) > config["batch_size"]:
                trader.batch_train()

            pbar.update()

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


def validate(label: str, writer: pd.ExcelWriter):
    # Initialize dataFrames (not adding columns dynamically due to performance)
    column_headers = []
    for i, sessions in enumerate(dp.batched_dir):
        column_headers.extend(f"b{i}s{n}" for n in range(len(sessions)))

    init_balances = [[config["initial_balance"]] * len(column_headers)] * (
        sequences_per_batch
    )
    init_actions = [["STAY"] * len(column_headers)] * (sequences_per_batch)

    balance_list = pd.DataFrame(init_balances, columns=column_headers)
    action_list = pd.DataFrame(init_actions, columns=column_headers)

    pbar = ProgressBar(
        episodes=1,
        batches=len(dp.batched_dir),
        sequences_per_batch=sequences_per_batch,
        prefix=f"Validation: {label}",
        leave=True,
    )

    for i in range(len(dp.batched_dir)):
        batch = dp.load_batch(i)
        states = init_states(len(batch[0]))

        for idx, sequences in enumerate(batch):
            for seq, state in zip(sequences, states):
                state.data = seq

            q_values = trader.predict(states)
            for ids, qs in enumerate(zip(q_values, states)):
                amount = action_space.calc_trade_amount(qs[0], qs[1])
                action = action_space.take_action(qs[0], qs[1])[1]
                action_list[f"b{i}s{ids}"].iloc[idx] = f"{action}|{amount}"
                balance_list[f"b{i}s{ids}"].iloc[idx] = qs[1].balance

            pbar.update()

    pbar.close()

    balance_list.to_excel(writer, sheet_name=f"{label}_balances", index=False)
    action_list.to_excel(writer, sheet_name=f"{label}_actions", index=False)


trader.memory.clear()
trader.epsilon = 0  # This removes random choices
writer = pd.ExcelWriter(
    f"{config['validation_dir']}/validation_{config['model_name']}_{now}.xlsx",
    engine="xlsxwriter",
)

validate("Training data", writer)
# Loading validation dataset
dp.dir = config["data_validation"]
dp.batched_dir = dp.batch_dir()
validate("Validation data", writer)

writer.close()
