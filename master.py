from src.data_processor import DataProcessor
from src.trader import FreeLaborTrader
from src.state import State
from src.action_space import ActionSpace
from src.progress_bar import ProgressBar
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
    sequence_length=config["sequence_length"],
    batch_size=config["batch_size"],
    headers=config["data_headers"],
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
)
now = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
trader.model.summary()


def calc_terminal_reward(reward: float, state: "State") -> float:
    if state.has_position or state.balance < 0:
        return -100000000000000000000
    else:
        return (reward + states.balance - config["initial_balance"]) * config[
            "reward_factors"
        ]["session_total"]


def rem_time(t: float, num_ep: int):
    rem_time_sec = num_ep * (time.time() - t)
    return (
        f"Remaining time: {round(rem_time_sec / 60)} min {round(rem_time_sec) % 60} sec"
    )


########################### Training ###########################

dp.dir = config["training_data"]
dp.batched_dir = dp.batch_dir()
terminal_model = (
    f"{config['model_directory']}/"
    + f"{config['model_name']}_"
    + f"{now}"
    + "_terminal.h5"
)
pbar = ProgressBar(
    episodes=config["episodes"],
    batches=len(dp.batched_dir),
    sequences_per_batch=len(dp.load_batch(0)),
    prefix="Training",
    suffix="Remaining time: ???",
    leave=True,
)

for e in range(1, config["episodes"] + 1):
    t1 = time.time()
    for i in range(len(dp.batched_dir)):
        batch = dp.load_batch(i)
        done = False

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
                rewards = [calc_terminal_reward(r, s) for r, s in zip(rewards, states)]

            for snap, reward, state in zip(snapshot, rewards, states.copy()):
                trader.memory.add((snap, reward, state, done))

            if len(trader.memory) > config["batch_size"]:
                trader.batch_train()

            pbar.update(e, i + 1, idx + 1)

        # Create hindsight experiences
        trader.memory.analyze_missed_opportunities(action_space)

    # Save the model every 10 episodes
    if e % 10 == 0:
        trader.model.save(
            f"{config['model_directory']}/{config['model_name']}_ep{e}_{now}.h5"
        )
    pbar.suffix = rem_time(t1, config["episodes"] + 1 - e)

trader.model.save(terminal_model)
pbar.close()

###################### Validation | Test #######################

dp.dir = config["validation_data"]
# dp.dir = config["test_data"]
dp.batched_dir = dp.batch_dir()
dp.step_size = 1
trader.memory.clear()
trader.load(terminal_model)

pbar = ProgressBar(
    episodes=1,
    batches=len(dp.batched_dir),
    sequences_per_batch=len(dp.load_batch(0)),
    prefix="Validation",
    suffix="Remaining time: ???",
    leave=True,
)

for i in range(len(dp.batched_dir) - 1):
    t1 = time.time()
    batch = dp.load_batch(i)

    # Initial states
    states = [State(data=empty_sequence(), balance=config["initial_balance"])] * config[
        "batch_size"
    ]

    for idx, sequences in enumerate(batch):
        for seq, state in zip(sequences, states):
            state.data = seq

        q_values = trader.predict(states)
        for q, s in zip(q_values, states):
            action_space.take_action(q, s)

        pbar.update(batch=i + 1, seq=idx)

    pbar.suffix = rem_time(t1, len(dp.batched_dir) - i)
