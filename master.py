from src.data_processor import DataProcessor
from src.trader import FreeLaborTrader
from src.state import State
from src.action_space import ActionSpace
import pandas as pd
import numpy as np
import yaml
from yaml.loader import FullLoader

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
    dir=config["data_directory"],
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
trader.model.summary()


def create_state(sequence: pd.DataFrame, state: "State" = None) -> "State":
    return (
        State(
            data=sequence,
            balance=state.balance,
            entry_price=state.entry_price,
            contracts=state.contracts,
        )
        if state
        else State(sequence, balance=config["initial_balance"])
    )


def calc_terminal_reward(reward: float, state: "State") -> float:
    if state.has_position or state.balance < 0:
        return -100000000000000000000
    else:
        return (reward + states.balance - config["initial_balance"]) * config[
            "reward_factors"
        ]["session_total"]


for i in range(len(dp.batched_dir) - 1):
    batch = dp.load_batch(i)

    for e in range(1, config["episodes"] + 1):
        done = False

        # Initial states
        states = [
            State(data=empty_sequence(), balance=config["initial_balance"])
        ] * config["batch_size"]

        for idx, sequences in enumerate(batch):
            if done:
                continue

            states = [
                State(
                    data=seq,
                    balance=state.balance,
                    entry_price=state.entry_price,
                    contracts=state.contracts,
                )
                for seq, state in zip(sequences, states)
            ]
            snapshot = states.copy()  # States before action; For experiences

            q_values = trader.predict(states)
            rewards = [action_space.take_action(q, s) for q, s in zip(q_values, states)]

            # Check for next state to be available
            done = idx + 1 == len(batch) - 1
            if done:
                rewards = [calc_terminal_reward(r, s) for r, s in zip(rewards, states)]

            for snap, reward, state in zip(snapshot, rewards, states):
                trader.memory.add((snap, reward, state, done))

            if len(trader.memory) > config["batch_size"]:
                trader.batch_train()

        # Create hindsight experiences
        trader.memory.analyze_missed_opportunities(action_space)

        # Save the model every 10 episodes
        if e % 10 == 0:
            trader.model.save(
                f"{config['model_directory']}/{config['model_name']}_ep{e}.h5"
            )

    trader.model.save(f"{config['model_directory']}/{config['model_name']}_terminal.h5")
