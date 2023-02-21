from src.data_processor import DataProcessor, Data
from src.trader import FreeLaborTrader
from src.state import State
import numpy as np
import types

from tqdm import tqdm


required_headers = ["Open", "High", "Low", "Close", "Volume"]
dropped_headers = ["DateTime"]

episodes = 100
batch_size = 6
sequence_length = 5
update_freq = 5
tick_size = 0.25
tick_value = 12.50

dp = DataProcessor(
    sequence_length=sequence_length,
    window_size=batch_size,
    column_headers=required_headers,
    dropped_headers=dropped_headers,
)
data = Data("data/ES_futures_sample/ES_continuous_1min_sample.csv", dp)
trader = FreeLaborTrader(
    sequence_length=sequence_length, batch_size=batch_size, num_features=8
)

trader.model.summary()

for episode in range(1, episodes + 1):
    print(f"Episode: {episode}/{episodes}")
    done = False

    # Initial parameters for each episode
    initial_balance = 10000  # Initial balance may change from episode to episode
    initial_entry_price = 0  # This should only change when swing trading
    initial_contracts = 0  # This should only change when swing trading

    state = State(
        data.sequenced[0],
        balance=initial_balance,
        entry_price=initial_entry_price,
        contracts=initial_contracts,
    )  # initial state

    # TODO: Determine prices
    current_price = data.raw["Close"].iloc[-1]

    for i, batch in enumerate(tqdm(data.windowed)):
        # Initialize states for batch of data; last element represents current state
        batch_states = [
            State(sequence, state.balance, state.entry_price, state.contracts)
            for sequence in batch
        ]
        state = batch_states[-1]

        next_sequence = data.get_next_sequence(batch[-1])
        if type(next_sequence) is types.NoneType:
            continue

        next_state = State(
            next_sequence,
            state.balance,
            state.entry_price,
            state.contracts,
        )

        # TODO: Revise the goal (e.g. absolute vs relative value)
        # Define the desired goal as the closing price of the next time step
        # desired_goal = next_state.close

        action = trader.predict_action(np.array([s.to_numpy() for s in batch_states]))
        reward = 0

        if action == 1 and not state.has_position():  # Buying; enter long position
            # TODO: Make possible to buy multiple contracts based on current balance
            next_state.enter_long(current_price, 1, tick_value)
            # print("FreeLaborTrader entered position:", state.rep_position())

        elif action == 2 and not state.has_position():  # Selling; enter short position
            # TODO: Make possible to sell short multiple contracts based on current balance
            next_state.enter_short(current_price, 1, tick_value)
            # print("FreeLaborTrader entered position:", state.rep_position())

        elif action == 3 and state.has_position():  # Exit; close position
            # TODO: Calculate actual profit
            # TODO: Calculate reward based on position exit
            profit = next_state.exit_position(current_price, tick_value)
            # print("FreeLaborTrader exited position with profit: ", profit)
            reward = profit

        done = (
            i == len(data.windowed) - 1
            or type(data.get_next_sequence(data.windowed[i + 1][-1])) is types.NoneType
        )
        if done:
            # Consequences for braking restrictions
            reward = (
                -1000000000000000
                if next_state.has_position() or next_state.balance < 0
                else reward
            )
            print("########################")
            print(f"TOTAL PROFIT: {next_state.balance - initial_balance}")
            print("########################")

            # Updating the initial values of each episode
            initial_balance = next_state.balance
            initial_entry_price = next_state.entry_price
            initial_contracts = next_state.contracts

        trader.memory.add(
            (
                state.to_numpy(),
                action,
                reward,
                next_state.to_numpy(),
                done,
            )
        )

        state = next_state
        current_price = batch[-1]["Close"].iloc[-1]

        if len(trader.memory) > batch_size:
            trader.batch_train()

    # Save the model every 10 episodes
    if episode % 10 == 0:
        trader.model.save(f"models/v0.1_ep{episode}.h5")
