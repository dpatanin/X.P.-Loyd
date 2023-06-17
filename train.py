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

with open("config.yaml") as f:
    config = yaml.load(f, Loader=FullLoader)


def num_features() -> int:
    return len(State(data=empty_sequence()).to_df().columns)


def empty_sequence() -> pd.DataFrame:
    empty = np.zeros((config["sequence_length"], len(config["data_headers"])))
    return pd.DataFrame(empty, columns=config["data_headers"])


action_space = ActionSpace(
    threshold=config["action_space"]["threshold"],
    price_per_contract=config["tick_value"],
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


def calc_terminal_reward(reward: float, state: "State") -> float:
    if state.has_position():
        reward += state.exit_position(config["tick_value"])
    if state.balance < 0:
        return reward - 10000000000000000000000
    else:
        return (reward + state.balance - config["initial_balance"]) * config[
            "reward_factors"
        ]["session_total"]


def rem_time(times: list[int], it_left: int):
    rem_time_sec = it_left * (sum(times) / len(times))
    return f"Remaining time: {math.floor(rem_time_sec / 3600)} h {math.floor(rem_time_sec / 60) % 60} min"


def avg_profit(states: list["State"]):
    sum_balance = sum(state.balance for state in states)
    return (sum_balance / config["batch_size"]) - config["initial_balance"]


########################### Training ###########################

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
profit_list = []
times_per_batch = []

for e in range(1, config["episodes"] + 1):
    for i in range(len(dp.batched_dir)):
        t = time.time()
        batch = dp.load_batch(i)

        # Initial states
        states = [
            State(data=empty_sequence(), balance=config["initial_balance"])
        ] * config["batch_size"]

        for idx, sequences in enumerate(batch):
            for seq, state in zip(sequences, states):
                state.data = seq
            snapshot = states.copy()  # States before action; For experiences

            q_values = trader.predict(states)
            rewards = [action_space.take_action(q, s) for q, s in zip(q_values, states)]

            done = idx == len(batch) - 1
            if done:
                profit_list.append(avg_profit(states))
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

    # Save the model to be served
    tf.saved_model.save(
        trader.model, f'/{config["model_directory"]}/{config["version"]}'
    )

    # Save copy in h5 format
    trader.model.save(
        f"{config['model_directory']}/{config['model_name']}_{'terminal' if e == config['episodes'] else f'ep{e}'}_{now}.h5"
    )

pbar.close()
df = pd.DataFrame(profit_list)
df.to_excel(f"data/training_{config['model_name']}_{now}.xlsx")
