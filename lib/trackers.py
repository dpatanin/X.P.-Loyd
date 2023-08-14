"""Components for statistics and Tensorboard monitoring."""
import collections
import contextlib
import shutil
import timeit
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Text, Tuple, Union

import numpy as np
import tensorflow as tf

import lib.replay as replay_lib
from lib.types import TimeStep


class EpisodeTracker:
    """Tracks episode return and other statistics."""

    def __init__(self):
        self._num_steps_since_reset = None
        self._episode_returns = None
        self._episode_steps = None
        self._episode_visited_rooms = None
        self._current_episode_rewards = None
        self._current_episode_step = None
        self._current_episode_profits = None
        self._current_episode_fees = None
        self._episode_profits = None
        self._episode_fees = None

    def step(self, env, timestep_t: TimeStep, agent, a_t) -> None:
        """Accumulates statistics from timestep."""
        del (env, agent, a_t)

        # First reward is invalid, all other rewards are appended.
        if timestep_t.first:
            if self._current_episode_rewards:
                raise ValueError("Current episode reward list should be empty.")
            if self._current_episode_step != 0:
                raise ValueError("Current episode step should be zero.")
        else:
            reward = timestep_t.reward
            self._current_episode_rewards.append(reward)

        self._num_steps_since_reset += 1
        self._current_episode_step += 1
        self._current_episode_profits.append(timestep_t.info["profit"])
        self._current_episode_fees.append(timestep_t.info["fees"])

        if timestep_t.done:
            self.append_episode_results()

    def append_episode_results(self):
        self._episode_returns.append(sum(self._current_episode_rewards))
        self._episode_steps.append(self._current_episode_step)
        self._current_episode_rewards = []
        self._current_episode_step = 0
        self._episode_profits.append(sum(self._current_episode_profits))
        self._episode_fees.append(sum(self._current_episode_fees))
        self._current_episode_profits = []
        self._current_episode_fees = []

    def reset(self) -> None:
        """Resets all gathered statistics, not to be called between episodes."""
        self._num_steps_since_reset = 0
        self._episode_returns = []
        self._episode_steps = []
        self._current_episode_step = 0
        self._current_episode_rewards = []
        self._current_episode_fees = []
        self._current_episode_profits = []
        self._episode_profits = []
        self._episode_fees = []

    def get(self) -> Mapping[str, Union[int, float, None]]:
        if len(self._episode_returns) > 0:
            mean_episode_return = np.array(self._episode_returns).mean()
        else:
            mean_episode_return = sum(self._current_episode_rewards)

        return {
            "mean_episode_return": mean_episode_return,
            "num_episodes": len(self._episode_returns),
            "current_episode_step": self._current_episode_step,
            "num_steps_since_reset": self._num_steps_since_reset,
            "avg_episode_profits": np.array(self._episode_profits).mean(),
            "avg_episode_fees": np.array(self._episode_fees).mean(),
        }


class StepRateTracker:
    """Tracks step rate, number of steps taken and duration since last reset."""

    def __init__(self):
        self._num_steps_since_reset = None
        self._start = None

    def step(self, env, timestep_t, agent, a_t) -> None:
        """Accumulates statistics from timestep."""
        del (env, timestep_t, agent, a_t)

        self._num_steps_since_reset += 1

    def reset(self) -> None:
        """Reset statistics."""
        self._num_steps_since_reset = 0
        self._start = timeit.default_timer()

    def get(self) -> Mapping[Text, float]:
        """Returns statistics as a dictionary."""
        duration = timeit.default_timer() - self._start
        if self._num_steps_since_reset > 0:
            step_rate = self._num_steps_since_reset / duration
        else:
            step_rate = np.nan
        return {
            "step_rate": step_rate,
            "num_steps_since_reset": self._num_steps_since_reset,
            "duration": duration,
        }


class TensorboardEpisodeTracker(EpisodeTracker):
    """Extend EpisodeTracker to write to tensorboard"""

    def __init__(self, writer: tf.summary.SummaryWriter):
        super().__init__()
        self._total_steps = 0  # keep track total number of steps, does not reset
        self._total_episodes = 0  # keep track total number of episodes, does not reset
        self._writer = writer

    def step(self, env, timestep_t: TimeStep, agent, a_t) -> None:
        super().step(env, timestep_t, agent, a_t)

        self._total_steps += 1

        # To improve performance, only logging at end of an episode.
        if timestep_t.done:
            self._total_episodes += 1
            tb_steps = self._total_steps

            # tracker per episode
            episode_return = self._episode_returns[-1]
            episode_step = self._episode_steps[-1]
            episode_profit = self._episode_profits[-1]
            episode_fees = self._episode_fees[-1]

            with self._writer.as_default():
                # tracker per step
                tf.summary.scalar(
                    "performance(env_steps)/num_episodes",
                    self._total_episodes,
                    tb_steps,
                )
                tf.summary.scalar(
                    "performance(env_steps)/episode_return", episode_return, tb_steps
                )
                tf.summary.scalar(
                    "performance(env_steps)/episode_steps", episode_step, tb_steps
                )
                tf.summary.scalar(
                    "performance(env_steps)/profit", episode_profit, tb_steps
                )
                tf.summary.scalar("performance(env_steps)/fees", episode_fees, tb_steps)


class TensorboardStepRateTracker(StepRateTracker):
    """Extend StepRateTracker to write to tensorboard, for single thread training agent only."""

    def __init__(self, writer: tf.summary.SummaryWriter):
        super().__init__()

        self._total_steps = 0  # keep track total number of steps, does not reset
        self._writer = writer

    def step(self, env, timestep_t: TimeStep, agent, a_t) -> None:
        """Accumulates statistics from timestep."""
        super().step(env, timestep_t, agent, a_t)

        self._total_steps += 1

        # To improve performance, only logging at end of an episode.
        if timestep_t.done:
            time_stats = self.get()
            with self._writer.as_default():
                tf.summary.scalar(
                    "performance(env_steps)/step_rate",
                    time_stats["step_rate"],
                    self._total_steps,
                )


class TensorboardAgentStatisticsTracker:
    """Write agent statistics to tensorboard"""

    def __init__(self, writer: tf.summary.SummaryWriter):
        self._total_steps = 0  # keep track total number of steps, does not reset
        self._writer = writer

    def step(self, env, timestep_t: TimeStep, agent, a_t) -> None:
        """Accumulates statistics from timestep."""
        del (env, a_t)
        self._total_steps += 1

        # To improve performance, only logging at end of an episode.
        # This should not block the training loop if there's any exception.
        if timestep_t.done:
            with contextlib.suppress(Exception), self._writer.as_default():
                if stats := agent.statistics:
                    for k, v in stats.items():
                        if isinstance(v, (int, float)):
                            tf.summary.scalar(
                                f"agent_statistics(env_steps)/{k}",
                                v,
                                self._total_steps,
                            )

    def reset(self) -> None:
        """Reset statistics."""
        pass

    def get(self) -> Mapping[Text, float]:
        """Returns statistics as a dictionary."""
        return {}


class TensorboardLearnerStatisticsTracker:
    """Write learner statistics to tensorboard, for parallel training agents with actor-learner scheme"""

    def __init__(self, writer: tf.summary.SummaryWriter):
        self._total_steps = 0  # keep track total number of steps, does not reset
        self._num_steps_since_reset = 0
        self._start = timeit.default_timer()
        self._writer = writer

    def step(self, stats) -> None:
        """Accumulates statistics from timestep."""

        self._total_steps += 1
        self._num_steps_since_reset += 1

        # Log every N learner steps.
        if self._total_steps % 100 == 0:
            time_stats = self.get()
            with self._writer.as_default():
                tf.summary.scalar(
                    "learner_statistics(learner_steps)/step_rate",
                    time_stats["step_rate"],
                    self._total_steps,
                )

                # This should not block the training loop if there's any exception.
                with contextlib.suppress(Exception):
                    if stats:
                        for k, v in stats.items():
                            if isinstance(v, (int, float)):
                                tf.summary.scalar(
                                    f"learner_statistics(learner_steps)/{k}",
                                    v,
                                    self._total_steps,
                                )

    def reset(self) -> None:
        """Reset statistics."""
        self._num_steps_since_reset = 0
        self._start = timeit.default_timer()

    def get(self) -> Mapping[Text, float]:
        """Returns statistics as a dictionary."""
        duration = timeit.default_timer() - self._start
        if self._num_steps_since_reset > 0:
            step_rate = self._num_steps_since_reset / duration
        else:
            step_rate = np.nan
        return {
            "step_rate": step_rate,
            "num_steps_since_reset": self._num_steps_since_reset,
            "duration": duration,
        }


def make_default_trackers(log_name=None):
    if not log_name:
        return [EpisodeTracker(), StepRateTracker()]
    log_name = Path(f"logs/{log_name}")

    # Remove existing log directory
    if log_name.is_dir():
        shutil.rmtree(log_name)

    writer = tf.summary.create_file_writer(str(log_name))

    return [
        TensorboardEpisodeTracker(writer),
        TensorboardStepRateTracker(writer),
        TensorboardAgentStatisticsTracker(writer),
    ]


def make_learner_trackers(run_log_dir=None):
    if not run_log_dir:
        return []
    tb_log_dir = Path(f"logs/{run_log_dir}")

    # Remove existing log directory
    if tb_log_dir.is_dir():
        shutil.rmtree(tb_log_dir)

    writer = tf.summary.create_file_writer(str(tb_log_dir))

    return [TensorboardLearnerStatisticsTracker(writer)]


def generate_statistics(
    trackers: Iterable[Any],
    timestep_action_sequence: Iterable[Tuple[Optional[replay_lib.Transition]]],
) -> Mapping[str, Any]:
    """Generates statistics from a sequence of timestep and actions."""
    # Only reset at the start, not between episodes.
    for tracker in trackers:
        tracker.reset()

    for env, timestep_t, agent, a_t in timestep_action_sequence:
        for tracker in trackers:
            tracker.step(env, timestep_t, agent, a_t)

    # Merge all statistics dictionaries into one.
    statistics_dicts = (tracker.get() for tracker in trackers)
    return dict(collections.ChainMap(*statistics_dicts))
