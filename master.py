from src.data_processor import DataProcessor
from src.trader import FreeLaborTrader
from src.state import State
from src.action_space import ActionSpace
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
trade_limit = 500  # Limit to trade at once

terminal_reward_fac = 1.5
intrinsic_reward_fac = 0.75
hindsight_reward_fac = 0.5

action_space = ActionSpace(
    threshold=threshold,
    price_per_contract=tick_value,
    limit=trade_limit,
    intrinsic_fac=1,
)
dp = DataProcessor(
    dir="data",
    sequence_length=sequence_length,
    batch_size=batch_size,
    headers=headers,
)
trader = FreeLaborTrader(
    sequence_length=sequence_length,
    batch_size=batch_size,
    num_features=8,
    update_freq=1,
    hindsight_reward_fac=hindsight_reward_fac,
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
                action_space.take_action(q, s, ns)
                for q, s, ns in zip(q_values, curr_states, next_states)
            ]

            # Check for next state to be available
            done = idx + 1 == len(batch) - 1
            if done:
                # Punish if breaking restrictions or reward total profit
                terminal_rewards: list[float] = [
                    -1000000000000000000
                    if ns.has_position() or ns.balance < 0
                    else (r + ns.balance - init_balance) * terminal_reward_fac
                    for r, ns in zip(rewards, next_states)
                ]

            con_states = next_states
            for s, q, r, ns in zip(curr_states, q_values, rewards, next_states):
                trader.memory.add((s, q, r, ns, done))

            if len(trader.memory) > batch_size:
                trader.batch_train()

        # Create hindsight experiences
        trader.memory.analyze_missed_opportunities(action_space)

        # Save the model every 10 episodes
        if e % 10 == 0:
            trader.model.save(f"models/v0.1_ep{e}.h5")

    trader.model.save("models/terminal_model.h5")
