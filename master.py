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
threshold = 0.2
volume_limit = 500  # TODO

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


def take_action(q: int, state: "State", current_price: float):
    reward = 0
    amount = abs(((abs(q) - threshold) / (1 - threshold)) * volume_limit)
    if q > 0 and (overhead := state.balance - (amount * tick_value)) < 0:
        amount -= overhead / tick_value

    if q < 0:
        if abs(q) < threshold:
            if state.has_position() > 0:
                reward = state.exit_position()
        elif state.has_position() > 0:
            reward = state.exit_position()
            state.enter_short(current_price, amount, tick_value)
        elif state.has_position() < 0:
            contracts = state.contracts
            state.exit_position()
            state.enter_short(current_price, amount + contracts, tick_value)
        else:
            state.enter_short(current_price, amount, tick_value)
    elif q > 0:
        if abs(q) < threshold:
            if state.has_position() < 0:
                reward = state.exit_position()
        elif state.has_position() < 0:
            reward = state.exit_position()
            state.enter_long(current_price, amount, tick_value)
        elif state.has_position() > 0:
            contracts = state.contracts
            state.exit_position()
            state.enter_long(current_price, amount + contracts, tick_value)
        else:
            state.enter_long(current_price, amount, tick_value)
    else:
        # TODO: intrinsic motivation?
        reward += 0

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

            q_values = trader.predict(curr_states)
            rewards = [
                take_action(q, ns, ns.data["Close"].iloc[-1])
                for q, ns in zip(q_values, next_states)
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
            for s, q, r, ns in zip(curr_states, q_values, rewards, next_states):
                trader.memory.add((s, q, r, ns, done))

            if len(trader.memory) > batch_size:
                trader.batch_train()

        # Create hindsight experiences
        trader.memory.analyze_missed_opportunities(tick_value)

        # Save the model every 10 episodes
        if e % 10 == 0:
            trader.model.save(f"models/v0.1_ep{e}.h5")
