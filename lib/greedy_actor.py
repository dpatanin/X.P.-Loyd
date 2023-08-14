"""Greedy actors for testing and evaluation."""
from typing import Mapping, Text

import numpy as np
import tensorflow as tf

import lib.types as types_lib
from lib.R2D2 import RnnDqnNetworkInputs


def apply_egreedy_policy(
    q_values: tf.Tensor,
    epsilon: float,
    random_state: np.random.RandomState,  # pylint: disable=no-member
) -> types_lib.Action:
    """Apply e-greedy policy."""
    action_dim = q_values.shape[-1]
    return (
        random_state.randint(0, action_dim)
        if random_state.rand() <= epsilon
        else q_values.math.argmax(-1).item()
    )


class EpsilonGreedyActor(types_lib.Agent):
    """DQN e-greedy actor."""

    def __init__(
        self,
        network: tf.Module,
        exploration_epsilon: float,
        random_state: np.random.RandomState,
        name: str = "DQN-greedy",
    ):
        self.agent_name = name
        self._network = network
        self._exploration_epsilon = exploration_epsilon
        self._random_state = random_state

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Give current timestep, return best action"""
        return self._select_action(timestep)

    def reset(self) -> None:
        """Resets the agent's episodic state such as frame stack and action repeat.
        This method should be called at the beginning of every episode.
        """

    def _select_action(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Samples action from eps-greedy policy wrt Q-values at given state."""
        s_t = tf.Tensor(timestep.observation[None, ...])
        q_t = self._network(s_t).q_values
        return apply_egreedy_policy(q_t, self._exploration_epsilon, self._random_state)

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        return {
            "exploration_epsilon": self._exploration_epsilon,
        }


class R2d2EpsilonGreedyActor(EpsilonGreedyActor):
    """R2D2 e-greedy actor."""

    def __init__(
        self,
        network: tf.Module,
        exploration_epsilon: float,
        random_state: np.random.RandomState,
    ):
        super().__init__(
            network,
            exploration_epsilon,
            random_state,
            "R2D2-greedy",
        )
        self._last_action = None
        self._lstm_state = None

    def _select_action(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Samples action from eps-greedy policy wrt Q-values at given state."""
        s_t = tf.Tensor(timestep.observation[None, ...]).to(
            device=self._device, dtype=tf.dtypes.float32
        )
        a_tm1 = tf.Tensor(self._last_action).to(
            device=self._device, dtype=tf.dtypes.int64
        )
        r_t = tf.Tensor(timestep.reward).to(
            device=self._device, dtype=tf.dtypes.float32
        )
        hidden_s = tuple(s.to(device=self._device) for s in self._lstm_state)

        network_output = self._network(
            RnnDqnNetworkInputs(
                s_t=s_t[None, ...],
                a_tm1=a_tm1[None, ...],
                r_t=r_t[None, ...],
                hidden_s=hidden_s,
            )
        )
        q_t = network_output.q_values
        self._lstm_state = network_output.hidden_s

        a_t = apply_egreedy_policy(q_t, self._exploration_epsilon, self._random_state)
        self._last_action = a_t
        return a_t

    def reset(self) -> None:
        """Reset hidden state to zeros at new episodes."""
        self._last_action = 0  # During the first step of a new episode, use 'fake' previous action for network pass
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)
