from src.data_preprocess import load_data
from src.trader import FreeLaborTrader
from src.state import State
import pandas as pd

from tqdm import tqdm


def state_creator(data: pd.DataFrame, timestep: int, state: State = None):
    new_data = data.iloc[[timestep]].to_dict()
    new_state = State(
        new_data["Open"][timestep],
        new_data["High"][timestep],
        new_data["Low"][timestep],
        new_data["Close"][timestep],
        new_data["Volume"][timestep],
    )

    if state:
        new_state.balance = state.balance
        new_state.entry_price = state.entry_price
        new_state.contracts = state.contracts

    return new_state


data = load_data("data/ES_futures_sample/ES_continuous_1min_sample.csv")
episodes = 100
batch_size = 32
data_samples = len(data) - 1
tick_size = 0.25
tick_value = 12.50
initial_balance = 10000

trader = FreeLaborTrader(state_size=8)
trader.model.summary()

for episode in range(1, episodes + 1):
    print(f"Episode: {episode}/{episodes}")
    state = state_creator(data, 0)

    # tqdm is used for visualization
    for t in tqdm(range(data_samples)):
        action = trader.trade(state.to_df().to_numpy())
        next_state = state_creator(data, t + 1, state)
        reward = 0

        if action == 1 and not state.has_position():  # Buying; enter long position
            # TODO: Make possible to buy multiple contracts based on current balance
            next_state.enter_long(state.close, 1, tick_value)
            # print("FreeLaborTrader entered position:", state.rep_position())

        elif action == 2 and not state.has_position():  # Selling; enter short position
            # TODO: Make possible to sell short multiple contracts based on current balance
            next_state.enter_short(state.close, 1, tick_value)
            # print("FreeLaborTrader entered position:", state.rep_position())

        elif action == 3 and state.has_position():  # Exit; close position
            # TODO: Calculate actual profit
            # TODO: Calculate reward based on position exit
            profit = next_state.exit_position(state.close, tick_value)
            # print("FreeLaborTrader exited position with profit: ", profit)
            reward = profit

        done = t == data_samples - 1
        if done:
            # Consequences for braking restrictions
            reward = (
                -1000000000000000
                if state.has_position() or state.balance < 0
                else reward
            )
            print("########################")
            print(f"TOTAL PROFIT: {state.balance - initial_balance}")
            print("########################")

        trader.memory.append(
            (
                state.to_df().to_numpy(),
                action,
                reward,
                next_state.to_df().to_numpy(),
                done,
            )
        )

        state = next_state

        if len(trader.memory) > batch_size:
            trader.batch_train(batch_size)

    # Save the model every 10 episodes
    if episode % 10 == 0:
        trader.model.save(f"models/v0.1_ep{episode}.h5")
