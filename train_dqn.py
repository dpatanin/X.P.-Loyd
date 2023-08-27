import json
import logging
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"  # or any {0:5}
warnings.simplefilter("ignore")

import keras
import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.networks import categorical_q_network
from tf_agents.policies import PolicySaver, random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tqdm import tqdm

from lib.data_processor import DataProcessor
from lib.trading_env import TFPyTradingEnvWrapper, TradingEnvironment

FEATURES = [
    "day_sin",
    "day_cos",
    "high_diff",
    "low_diff",
    "open",
    "close",
    "volume",
]

LOG_DIR = "logs"
MODEL_DIR = "models"
FULL_DATA = "https://onedrive.live.com/download?resid=2ba9b2044e887de1%21290022&authkey=!ADgq6YFliQNylSM"  # No sentiment but ~15 years
SENTIMENT_DATA = "https://onedrive.live.com/download?resid=2ba9b2044e887de1%21293628&authkey=!ANbFvs1RrC9WQ3c"  # With sentiment but ~5 years

SEQ_LENGTH = 15
BATCH_SIZE = 512
TRAIN_STEPS = 10000
EVAL_MIN_STEPS = 10000
CHECKPOINT_INTERVAL = 15000  # Agent
LOAD_CHECKPOINT = False
N_STEP_UPDATE = 2

dp = DataProcessor(FULL_DATA, 5)
pb = tqdm(range(5), desc="Create environments")


def update_pb(desc: str = None):
    pb.update()
    if desc:
        pb.set_description(desc)


def env_creator(df: pd.DataFrame, load_checkpoint: bool):
    return TFPyTradingEnvWrapper(
        TradingEnvironment(
            df=df,
            window_size=SEQ_LENGTH,
            features=FEATURES,
            balance=1000.00,
            fees_per_contract=0.25,
            streak_span=SEQ_LENGTH,
            streak_bonus_max=2,
            streak_difficulty=18,
            env_state_dir=json.load(open("logs/train")) if load_checkpoint else None,
        )
    )


train_env = env_creator(dp.train_df, LOAD_CHECKPOINT)
eval_env = env_creator(dp.val_df, False)

update_pb("Create network")
categorical_q_net = categorical_q_network.CategoricalQNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    num_atoms=51,
    fc_layer_params=(100,),
)


update_pb("Create agent")
optimizer = keras.optimizers.Nadam(learning_rate=1e-3)
train_step_counter = tf.Variable(0)

agent = categorical_dqn_agent.CategoricalDqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    categorical_q_network=categorical_q_net,
    optimizer=optimizer,
    min_q_value=-20,
    max_q_value=20,
    n_step_update=N_STEP_UPDATE,
    td_errors_loss_fn=common.element_wise_squared_loss,
    gamma=0.99,
    train_step_counter=train_step_counter,
)
agent.initialize()


update_pb("Initialize replay buffer")
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=100000,
)


def collect_step(environment, policy):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)


random_policy = random_tf_policy.RandomTFPolicy(
    train_env.time_step_spec(), train_env.action_spec()
)

for _ in range(1000):
    collect_step(train_env, random_policy)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=BATCH_SIZE, num_steps=N_STEP_UPDATE + 1
).prefetch(3)

iterator = iter(dataset)
agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)


update_pb("Load checkpointer")
train_checkpointer = common.Checkpointer(
    ckpt_dir=f"{MODEL_DIR}/checkpoints",
    max_to_keep=20,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
)
if LOAD_CHECKPOINT:
    if train_checkpointer.checkpoint_exists:
        try:
            train_checkpointer.initialize_or_restore()
        except Exception as error:
            logging.error(f"Checkpoint could not be restored: {error}")
    else:
        logging.error("No checkpoint found.")

update_pb("Training prepared!")
pb.close()


start_step = agent.train_step_counter.numpy()
try:
    with tqdm(range(TRAIN_STEPS), desc="Training") as pbar:
        for _ in range(TRAIN_STEPS):
            # Collect a few steps using collect_policy and save to the replay buffer.
            collect_step(train_env, agent.collect_policy)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience)
            step = agent.train_step_counter.numpy()

            if step % CHECKPOINT_INTERVAL == 0:
                train_checkpointer.save(step)

            pbar.set_description(f"Training | Step: {step} | Loss: {train_loss.loss}")
            pbar.update()

    with tqdm(range(EVAL_MIN_STEPS), desc="Evaluation") as pbar:
        time_step = eval_env.reset()
        eval_step = 0
        # not time_step.is_last() or
        while eval_step < EVAL_MIN_STEPS:
            eval_step += 1
            action_step = agent.policy.action(time_step)
            time_step = eval_env.step(action_step.action)
            pbar.update()

        eval_env.save(f"{LOG_DIR}/eval")

except KeyboardInterrupt:
    pass
except Exception as error:
    logging.error(error.with_traceback())
finally:
    try:
        train_env.save(f"{LOG_DIR}/train")
    except Exception as error:
        logging.error(error)

    train_checkpointer.save(step)
    PolicySaver(agent.policy).save(f"{MODEL_DIR}/final_{start_step}-{step}")
