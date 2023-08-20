import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"  # or any {0:5}
warnings.simplefilter("ignore")
import json
import logging

import pandas as pd
import reverb  # type: ignore
import tensorflow as tf
from tf_agents.agents.cql import cql_sac_agent
from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.policies import py_tf_eager_policy, random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.train import actor, learner, triggers
from tf_agents.train.utils import spec_utils, strategy_utils, train_utils
from tf_agents.utils import common
from tqdm import tqdm

from lib.data_processor import DataProcessor
from lib.trading_env import PyTradingEnvWrapper, TradingEnvironment

features = [
    "day_sin",
    "day_cos",
    "high_diff",
    "low_diff",
    "open_pct",
    "open_ema",
    "close_pct",
    "close_ema",
    "volume",
]

LOG_DIR = "logs"
MODEL_DIR = "models"
FULL_DATA = "https://onedrive.live.com/download?resid=2ba9b2044e887de1%21290022&authkey=!ADgq6YFliQNylSM"  # No sentiment but ~15 years
SENTIMENT_DATA = "https://onedrive.live.com/download?resid=2ba9b2044e887de1%21293628&authkey=!ANbFvs1RrC9WQ3c"  # With sentiment but ~5 years

SEQ_LENGTH = 30
BATCH_SIZE = 512
TIME_STEPS = 100000
CHECKPOINT_INTERVAL = 5000  # Agent
LOAD_CHECKPOINT = True
CHECKPOINT_LENGTH = 23 * 60  # Environment

dp = DataProcessor(FULL_DATA, 5)
pb = tqdm(range(16), desc="Create environments")


def update_pb(desc: str = None):
    pb.update()
    if desc:
        pb.set_description(desc)


def env_creator(df: pd.DataFrame):
    return PyTradingEnvWrapper(
        TradingEnvironment(
            df=df,
            window_size=SEQ_LENGTH,
            features=features,
            balance=10000.00,
            fees_per_contract=0.25,
            trade_limit=50,
            episode_history=json.load(open("logs/episode-history/100000-508441.json"))
            if LOAD_CHECKPOINT
            else None,
            checkpoint_length=CHECKPOINT_LENGTH if LOAD_CHECKPOINT else None,
        )
    )


collect_env = env_creator(dp.train_df)
eval_env = env_creator(dp.val_df)

update_pb("Get specs")
strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)

observation_spec, action_spec, time_step_spec = spec_utils.get_tensor_specs(collect_env)

with strategy.scope():
    update_pb("Create critic net")
    critic_net = critic_rnn_network.CriticRnnNetwork(
        (observation_spec, action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        lstm_size=[64],
        joint_fc_layer_params=(256, 256),
        kernel_initializer="glorot_uniform",
        last_kernel_initializer="glorot_uniform",
    )

    update_pb("Create actor net")
    actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
        observation_spec,
        action_spec,
        lstm_size=[64],
        continuous_projection_net=(
            tanh_normal_projection_network.TanhNormalProjectionNetwork
        ),
    )

    train_step = train_utils.create_train_step()

    update_pb("Initialize cql-sac agent")
    tf_agent = cql_sac_agent.CqlSacAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        critic_optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        alpha_optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        cql_alpha=0.2,
        num_cql_samples=5,
        target_update_tau=0.005,
        target_update_period=1,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=0.99,
        gradient_clipping=1,
        reward_scale_factor=1.0,
        train_step_counter=train_step,
        include_critic_entropy_term=False,
        use_lagrange_cql_alpha=True,
    )

    tf_agent.initialize()

update_pb("Create reverb replay")
rate_limiter = reverb.rate_limiters.SampleToInsertRatio(
    samples_per_insert=3.0, min_size_to_sample=3, error_buffer=3.0
)

table_name = "uniform_table"
table = reverb.Table(
    table_name,
    max_size=10000,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
)

reverb_server = reverb.Server([table])

reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
    tf_agent.collect_data_spec,
    sequence_length=2,
    table_name="uniform_table",
    local_server=reverb_server,
)

update_pb("Prefetch experiences")
dataset = reverb_replay.as_dataset(sample_batch_size=BATCH_SIZE, num_steps=2).prefetch(
    50
)
experience_dataset_fn = lambda: dataset

update_pb("Create policies")
tf_eval_policy = tf_agent.policy
eval_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_eval_policy, use_tf_function=True)
update_pb()

tf_collect_policy = tf_agent.collect_policy
collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
    tf_collect_policy, use_tf_function=True
)
update_pb()

random_policy = random_py_policy.RandomPyPolicy(
    collect_env.time_step_spec(), collect_env.action_spec()
)
update_pb()

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    reverb_replay.py_client, table_name, sequence_length=2, stride_length=1
)

update_pb("Run initial actor")
initial_collect_actor = actor.Actor(
    collect_env,
    random_policy,
    train_step,
    steps_per_run=10000,
    observers=[rb_observer],
)
initial_collect_actor.run()

update_pb("Create collect actor")
env_step_metric = py_metrics.EnvironmentSteps()
collect_actor = actor.Actor(
    collect_env,
    collect_policy,
    train_step,
    steps_per_run=1,
    metrics=actor.collect_metrics(10),
    summary_dir=os.path.join(LOG_DIR, "collect"),
    observers=[rb_observer, env_step_metric],
)

update_pb("Create eval actor")
eval_actor = actor.Actor(
    eval_env,
    eval_policy,
    train_step,
    episodes_per_run=20,
    metrics=actor.eval_metrics(20),
    summary_dir=os.path.join(LOG_DIR, "eval"),
)

update_pb("Create learner")
agent_learner = learner.Learner(
    MODEL_DIR,
    train_step,
    tf_agent,
    experience_dataset_fn,
    triggers=[triggers.StepPerSecondLogTrigger(train_step, interval=1000)],
    strategy=strategy,
    summary_root_dir=LOG_DIR,
)


def get_eval_metrics():
    eval_actor.run()
    return {metric.name: metric.result() for metric in eval_actor.metrics}


def log_eval_metrics(step, metrics):
    eval_results = (", ").join(
        "{} = {:.6f}".format(name, result) for name, result in metrics.items()
    )
    print("step = {0}: {1}".format(step, eval_results))


update_pb("Load checkpointer")
train_checkpointer = common.Checkpointer(
    ckpt_dir=f"{MODEL_DIR}/{learner.POLICY_CHECKPOINT_DIR}",
    max_to_keep=20,
    agent=tf_agent,
    policy=tf_agent.policy,
    replay_buffer=reverb_replay,
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

returns = []
start_step = agent_learner.train_step_numpy
try:
    with tqdm(range(TIME_STEPS), desc="Training") as pbar:
        for _ in range(TIME_STEPS):
            # Training.
            collect_actor.run()
            loss_info = agent_learner.run(iterations=1)

            # Evaluating.
            step = agent_learner.train_step_numpy

            if step % 10000 == 0:
                pbar.set_description("Evaluating")
                metrics = get_eval_metrics()
                log_eval_metrics(step, metrics)
                returns.append(metrics["AverageReturn"])

            if step % CHECKPOINT_INTERVAL == 0:
                train_checkpointer.save(step)

            pbar.set_description(
                f"Training | Step: {step} | Loss: {loss_info.loss.numpy()}"
            )

            pbar.update()

except KeyboardInterrupt:
    pass
except Exception as error:
    logging.error(error)
finally:
    try:
        collect_env.save_episode_history(
            f"{LOG_DIR}/episode-history/{start_step}-{step}"
        )
    except Exception as error:
        logging.error(error)

    train_checkpointer.save(step)
    tf.saved_model.save(tf_agent.policy, f"{MODEL_DIR}/final_{start_step}-{step}")
    rb_observer.close()
    reverb_server.stop()
