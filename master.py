from src.data_processor import DataProcessor
from src.trader import FreeLaborTrader
from src.state import State

from tqdm import tqdm


episodes = 100
batch_size = 32
tick_size = 0.25
tick_value = 12.50
initial_balance = 10000

dp = DataProcessor("data/ES_futures_sample/ES_continuous_1min_sample.csv", batch_size)
trader = FreeLaborTrader(state_size=8)

trader.model.summary()

for episode in range(1, episodes + 1):
    print(f"Episode: {episode}/{episodes}")
    state = State(dp.windowed_data[0], balance=initial_balance)  # initial state
    current_price = dp.windowed_data[0]["Close"].iloc[-1]

    # tqdm is used for visualization
    for batch in tqdm(dp.windowed_data[1:]):
        next_state = State(batch, state.balance, state.entry_price, state.contracts)

        # TODO: Revise the goal (e.g. absolute vs relative value)
        # Define the desired goal as the closing price of the next time step
        # desired_goal = next_state.close

        action = trader.trade(state.to_numpy())
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


        done = batch.index[-1] == len(dp.windowed_data) - 1
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
        current_price = batch["Close"].iloc[-1]

        if len(trader.memory) > batch_size:
            trader.batch_train(batch_size)

    # Save the model every 10 episodes
    if episode % 10 == 0:
        trader.model.save(f"models/v0.1_ep{episode}.h5")
