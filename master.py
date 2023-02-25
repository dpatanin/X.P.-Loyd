from src.data_processor import DataProcessor
from src.trader import FreeLaborTrader
from src.state import State
import pandas as pd


headers = ["Open", "High", "Low", "Close", "Volume"]

episodes = 100
batch_size = 4
sequence_length = 5
update_freq = 5
tick_size = 0.25
tick_value = 12.50
init_balance = 10000.00

dp = DataProcessor(
    dir="data",
    sequence_length=sequence_length,
    batch_size=batch_size,
    headers=headers,
)
trader = FreeLaborTrader(
    sequence_length=sequence_length, batch_size=batch_size, num_features=8
)
trader.model.summary()


def create_state(sequence: pd.DataFrame, state: "State" = None):
    return (
        State(
            data=sequence,
            balance=state.balance,
            entry_price=state.entry_price,
            contracts=state.contracts,
        )
        if state
        else State(sequence, balance=init_balance)
    )


def take_action(action: int, state: "State", current_price: float):
    reward = 0

    if action == 1 and not state.has_position():  # Buying; enter long position
        # TODO: Make possible to buy multiple contracts based on current balance
        state.enter_long(current_price, contracts=1, price_per_contract=tick_value)

    elif action == 2 and not state.has_position():  # Selling; enter short position
        # TODO: Make possible to sell short multiple contracts based on current balance
        state.enter_short(current_price, contracts=1, price_per_contract=tick_value)

    elif action == 3 and state.has_position():  # Exit; close position
        profit = state.exit_position(current_price, price_per_contract=tick_value)
        reward = profit

    return reward


for i in range(len(dp.batched_dir) - 1):
    batch = dp.load_batch(i)

    for e in range(1, episodes + 1):
        done = False

        # States to maintain continuity of actions
        con_states: list["State"] = [None] * batch_size

        for idx, sequences in enumerate(batch):
            if done:
                continue

            curr_states = [
                create_state(seq, state) for seq, state in zip(sequences, con_states)
            ]
            next_states = [
                create_state(seq, state)
                for seq, state in zip(batch[idx + 1], con_states)
            ]

            actions = trader.predict_actions(curr_states)
            rewards = [
                take_action(a, ns, ns.data["Close"].iloc[-1])
                for a, ns in zip(actions, next_states)
            ]

            # Check for next state to be available
            done = idx + 1 == len(batch) - 1
            if done:
                # Punish if breaking restrictions or reward total profit
                terminal_rewards: list[float] = [
                    -1000000000000000000
                    if ns.has_position() or ns.balance < 0
                    else r + ns.balance - init_balance
                    for r, ns in zip(rewards, next_states)
                ]

            con_states = next_states
            for s, a, r, ns in zip(curr_states, actions, rewards, next_states):
                trader.memory.add((s, a, r, ns, done))

            if len(trader.memory) > batch_size:
                trader.batch_train()

        # Create hindsight experiences
        trader.memory.analyze_missed_opportunities(tick_value)

        # Save the model every 10 episodes
        if e % 10 == 0:
            trader.model.save(f"models/v0.1_ep{e}.h5")
