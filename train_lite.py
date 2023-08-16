import os
import tempfile
from datetime import datetime

import pandas as pd
import reverb  # type: ignore
import tensorflow as tf
from tqdm import tqdm
from tf_agents.agents.cql import cql_sac_agent
from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.policies import py_tf_eager_policy, random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.train import actor, learner, triggers
from tf_agents.train.utils import spec_utils, strategy_utils, train_utils

from lib.data_processor import DataProcessor
from lib.trading_env import PyTradingEnvWrapper, TradingEnvironment

tempdir = tempfile.gettempdir()

now = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
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

DESC = "CQL-SAC"
FULL_DATA = "https://onedrive.live.com/download?resid=2ba9b2044e887de1%21290022&authkey=!ADgq6YFliQNylSM"  # No sentiment but ~15 years
SENTIMENT_DATA = "https://onedrive.live.com/download?resid=2ba9b2044e887de1%21293628&authkey=!ANbFvs1RrC9WQ3c"  # With sentiment but ~5 years
SEQ_LENGTH = 30
BATCH_SIZE = 512


dp = DataProcessor("source.csv", 5)


def env_creator(df: pd.DataFrame):
    return PyTradingEnvWrapper(
        TradingEnvironment(
            df=df,
            window_size=SEQ_LENGTH,
            features=features,
        )
    )


collect_env = env_creator(dp.train_df)
eval_env = env_creator(dp.val_df)

strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)

observation_spec, action_spec, time_step_spec = spec_utils.get_tensor_specs(collect_env)

with strategy.scope():
    critic_net = critic_rnn_network.CriticRnnNetwork(
        (observation_spec, action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        lstm_size=[64],
        joint_fc_layer_params=(256, 256),
        kernel_initializer="glorot_uniform",
        last_kernel_initializer="glorot_uniform",
    )

    actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
        observation_spec,
        action_spec,
        lstm_size=[64],
        continuous_projection_net=(
            tanh_normal_projection_network.TanhNormalProjectionNetwork
        ),
    )

    train_step = train_utils.create_train_step()

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

    dataset = reverb_replay.as_dataset(
        sample_batch_size=BATCH_SIZE, num_steps=2
    ).prefetch(50)
    experience_dataset_fn = lambda: dataset

    tf_eval_policy = tf_agent.policy
    eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_eval_policy, use_tf_function=True
    )

    tf_collect_policy = tf_agent.collect_policy
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_collect_policy, use_tf_function=True
    )

    random_policy = random_py_policy.RandomPyPolicy(
        collect_env.time_step_spec(), collect_env.action_spec()
    )

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        reverb_replay.py_client, table_name, sequence_length=2, stride_length=1
    )

    initial_collect_actor = actor.Actor(
        collect_env,
        random_policy,
        train_step,
        steps_per_run=10000,
        observers=[rb_observer],
    )
    initial_collect_actor.run()

    env_step_metric = py_metrics.EnvironmentSteps()
    collect_actor = actor.Actor(
        collect_env,
        collect_policy,
        train_step,
        steps_per_run=1,
        metrics=actor.collect_metrics(10),
        summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
        observers=[rb_observer, env_step_metric],
    )

    eval_actor = actor.Actor(
        eval_env,
        eval_policy,
        train_step,
        episodes_per_run=20,
        metrics=actor.eval_metrics(20),
        summary_dir=os.path.join(tempdir, "eval"),
    )

    saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

    # Triggers to save the agent's policy checkpoints.
    learning_triggers = [
        triggers.PolicySavedModelTrigger(
            saved_model_dir, tf_agent, train_step, interval=5000
        ),
        triggers.StepPerSecondLogTrigger(train_step, interval=1000),
    ]

    agent_learner = learner.Learner(
        tempdir,
        train_step,
        tf_agent,
        experience_dataset_fn,
        triggers=learning_triggers,
        strategy=strategy,
    )

    def get_eval_metrics():
        eval_actor.run()
        return {metric.name: metric.result() for metric in eval_actor.metrics}

    metrics = get_eval_metrics()

    def log_eval_metrics(step, metrics):
        eval_results = (", ").join(
            "{} = {:.6f}".format(name, result) for name, result in metrics.items()
        )
        print("step = {0}: {1}".format(step, eval_results))

    log_eval_metrics(0, metrics)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = get_eval_metrics()["AverageReturn"]
    returns = [avg_return]

    for _ in tqdm(range(100000)):
        # Training.
        collect_actor.run()
        loss_info = agent_learner.run(iterations=1)

        # Evaluating.
        step = agent_learner.train_step_numpy

        if step % 10000 == 0:
            metrics = get_eval_metrics()
            log_eval_metrics(step, metrics)
            returns.append(metrics["AverageReturn"])

        if step % 5000 == 0:
            print("step = {0}: loss = {1}".format(step, loss_info.loss.numpy()))

    rb_observer.close()
    reverb_server.stop()
