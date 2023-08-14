"""R2D2 agent class.

From the paper "Recurrent Experience Replay in Distributed Reinforcement Learning"
https://openreview.net/pdf?id=r1lyTjAqYX.

The code for value function rescaling, inverse value function rescaling, and n-step bellman targets are from seed-rl:
https://github.com/google-research/seed_rl/blob/66e8890261f09d0355e8bf5f1c5e41968ca9f02b/agents/r2d2/learner.py

This agent supports store hidden state (only first step in a unroll) in replay, and burn in.
In fact, even if we use burn in, we're still going to store the hidden state (only first step in a unroll) in the replay.
"""
import copy
import multiprocessing
from typing import Iterable, Mapping, NamedTuple, Optional, Text, Tuple

import keras
import numpy as np
import tensorflow as tf

import lib.assertions as assertions_lib
import lib.replay as replay_lib
import lib.types as types_lib
import lib.utils as utils

tf.config.run_functions_eagerly(True)

HiddenState = Tuple[tf.Tensor, tf.Tensor]


class RnnDqnNetworkInputs(NamedTuple):
    s_t: tf.Tensor
    a_tm1: tf.Tensor
    r_t: tf.Tensor  # reward for (s_tm1, a_tm1), but received at current timestep.
    hidden_s: Optional[Tuple[tf.Tensor, tf.Tensor]]


class RnnDqnNetworkOutputs(NamedTuple):
    q_values: tf.Tensor
    hidden_s: Optional[Tuple[tf.Tensor, tf.Tensor]]


class R2d2Transition(NamedTuple):
    """
    s_t, r_t, done are the tuple from env.step().

    last_action is the last agent the agent took, before in s_t.
    """

    s_t: Optional[np.ndarray]
    r_t: Optional[float]
    done: Optional[bool]
    a_t: Optional[int]
    q_t: Optional[np.ndarray]  # q values for s_t
    last_action: Optional[int]
    init_h: Optional[np.ndarray]  # nn.LSTM initial hidden state
    init_c: Optional[np.ndarray]  # nn.LSTM initial cell state


TransitionStructure = R2d2Transition(
    s_t=None,
    r_t=None,
    done=None,
    a_t=None,
    q_t=None,
    last_action=None,
    init_h=None,
    init_c=None,
)


def calculate_losses_and_priorities(
    q_value: tf.Tensor,
    action: tf.Tensor,
    reward: tf.Tensor,
    done: tf.Tensor,
    target_q_value: tf.Tensor,
    target_action: tf.Tensor,
    gamma: float,
    n_step: int,
    eps: float = 0.001,
    eta: float = 0.9,
) -> Tuple[tf.Tensor, tf.Tensor]:
    r"""Calculate loss and priority for given samples.

    T is the unrolled length, B the batch size, N is number of actions.

    Args:
        q_value: (T+1, B, action_dim) the predicted q values for a given state 's_t' from online Q network.
        action: [T+1, B] the actual action the agent take in state 's_t'.
        reward: [T+1, B] the reward the agent received at timestep t, this is for (s_tm1, a_tm1).
        done: [T+1, B] terminal mask for timestep t, state 's_t'.
        target_q_value: (T+1, B, N) the estimated TD n-step target values from target Q network,
            this could also be the same q values when just calculate priorities to insert into replay.
        target_action: [T+1, B] the best action to take in t+n timestep target state.
        gamma: discount rate.
        n_step: TD n-step size.
        eps: constant for value function rescaling and inverse value function rescaling.
        eta: constant for calculate mixture priorities.

    Returns:
        losses: the losses for given unrolled samples, shape (B, )
        priorities: the priority for given samples, shape (B, )
    """

    assertions_lib.assert_rank_and_dtype(q_value, 3, tf.dtypes.float32)
    assertions_lib.assert_rank_and_dtype(target_q_value, 3, tf.dtypes.float32)
    assertions_lib.assert_rank_and_dtype(reward, 2, tf.dtypes.float32)
    assertions_lib.assert_rank_and_dtype(action, 2, tf.dtypes.long)
    assertions_lib.assert_rank_and_dtype(target_action, 2, tf.dtypes.long)
    assertions_lib.assert_rank_and_dtype(done, 2, tf.dtypes.bool)

    q_value = q_value.gather(-1, action[..., None]).squeeze(-1)  # [T, B]

    target_q_max = target_q_value.gather(-1, target_action[..., None]).squeeze(
        -1
    )  # [T, B]
    # Apply invertible value rescaling to TD target.
    target_q_max = utils.signed_parabolic(target_q_max, eps)

    # Note the input rewards into 'n_step_bellman_target' should be non-discounted, non-summed.
    target_q = utils.n_step_bellman_target(
        r_t=reward, done=done, q_t=target_q_max, gamma=gamma, n_steps=n_step
    )

    # q_value is actually Q(s_t, a_t), but rewards is for 's_tm1', 'a_tm1',
    # that means our 'target_q' value is one step behind 'q_value',
    # so we need to shift them to make it in the same timestep.
    q_value = q_value[:-1, ...]
    target_q = target_q[1:, ...]

    # Apply value rescaling to TD target.
    target_q = utils.signed_hyperbolic(target_q, eps)

    if q_value.shape != target_q.shape:
        raise RuntimeError(
            f"Expect q_value and target_q have the same shape, got {q_value.shape} and {target_q.shape}"
        )

    td_error = target_q - q_value

    with tf.GradientTape(persistent=True) as tape:
        # Calculate priorities
        priorities = utils.calculate_dist_priorities_from_td_error(td_error, eta)

        # Sums over time dimension
        losses = 0.5 * tf.reduce_sum(tf.square(td_error), axis=0)  # [B]

    tf.debugging.check_numerics(losses, "Loss has NaNs or Inf.")

    del tape

    return losses, priorities


class Actor(types_lib.Agent):
    """R2D2 actor"""

    def __init__(
        self,
        data_queue: multiprocessing.Queue,
        network: tf.Module,
        random_state: np.random.RandomState,
        action_dim: int,
        unroll_length: int,
        burn_in: int,
        epsilon: float,
        actor_update_interval: int,
        shared_params: dict,
    ) -> None:
        """
        Args:
            data_queue: a multiprocessing.Queue to send collected transitions to learner process.
            network: the Q network for actor to make action choice.
            random_state: used to sample random actions for e-greedy policy.
            action_dim: the number of valid actions in the environment.
            unroll_length: how many agent time step to unroll transitions before put on to queue.
            burn_in: two consecutive unrolls will overlap on burn_in+1 steps. [0 `unroll_length`]
            epsilon: starting exploration factor.
            actor_update_interval: the frequency to update actor local Q network.
            shared_params: a shared dict, so we can later update the parameters for actors.
        """

        self.agent_name = "R2D2-actor"
        self._network = network

        # Disable autograd for actor's network
        self._network.set_trainable(False)

        self._shared_params = shared_params
        self._queue = data_queue
        self._random_state = random_state
        self._action_dim = action_dim
        self._actor_update_interval = actor_update_interval

        self._unroll = replay_lib.Unroll(
            unroll_length=unroll_length,
            overlap=burn_in + 1,  # Plus 1 to add room for shift during learning
            structure=TransitionStructure,
            cross_episode=False,
        )

        self._exploration_epsilon = epsilon

        self._last_action = None
        self._lstm_state = None  # Stores nn.LSTM hidden state and cell state

        self._step_t = -1

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Given timestep, return action a_t, and push transition into global queue"""
        self._step_t += 1

        if self._step_t % self._actor_update_interval == 0:
            self._update_actor_network()

        q_t, a_t, hidden_s = self.act(timestep)

        # Note the reward is for s_tm1, a_tm1, because it's only available one agent step after,
        # and the done mark is for current timestep s_t.
        transition = R2d2Transition(
            s_t=timestep.observation,
            a_t=a_t,
            q_t=q_t,
            r_t=timestep.reward,
            done=timestep.done,
            last_action=self._last_action,
            init_h=self._lstm_state[0].squeeze(1).numpy(),
            init_c=self._lstm_state[1].squeeze(1).numpy(),
        )

        unrolled_transition = self._unroll.add(transition, timestep.done)
        self._last_action, self._lstm_state = a_t, hidden_s

        if unrolled_transition is not None:
            self._put_unroll_onto_queue(unrolled_transition)

        return a_t

    def reset(self) -> None:
        """This method should be called at the beginning of every episode before take any action."""
        self._unroll.reset()
        self._last_action = self._random_state.randint(
            0, self._action_dim
        )  # Initialize a_tm1 randomly
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)

    def act(
        self, timestep: types_lib.TimeStep
    ) -> Tuple[np.ndarray, types_lib.Action, Tuple[tf.Tensor]]:
        "Given state s_t and done marks, return an action."
        return self._choose_action(timestep, self._exploration_epsilon)

    def _choose_action(
        self, timestep: types_lib.TimeStep, epsilon: float
    ) -> Tuple[np.ndarray, types_lib.Action, Tuple[tf.Tensor]]:
        """Given state s_t, choose action a_t"""
        pi_output = self._network(self._prepare_network_input(timestep))
        q_t = pi_output.q_values.squeeze()
        a_t = tf.math.argmax(q_t, axis=-1).cpu().item()

        # To make sure every actors generates the same amount of samples, we apply e-greedy after the network pass,
        # otherwise the actor with higher epsilons will generate more samples,
        # while the actor with lower epsilon will generate less samples.
        if self._random_state.rand() <= epsilon:
            # randint() return random integers from low (inclusive) to high (exclusive).
            a_t = self._random_state.randint(0, self._action_dim)

        return (q_t.cpu().numpy(), a_t, pi_output.hidden_s)

    def _prepare_network_input(
        self, timestep: types_lib.TimeStep
    ) -> RnnDqnNetworkInputs:
        # R2D2 network expect input shape [T, B, state_shape],
        # and additionally 'last action', 'reward for last action', and hidden state from previous timestep.
        s_t = tf.convert_to_tensor(timestep.observation[None, ...], dtype=tf.float32)
        a_tm1 = tf.convert_to_tensor(self._last_action, dtype=tf.int64)
        r_t = tf.convert_to_tensor(timestep.reward, dtype=tf.float32)
        hidden_s = tuple(
            tf.convert_to_tensor(s, dtype=tf.float32) for s in self._lstm_state
        )

        return RnnDqnNetworkInputs(
            s_t=s_t,  # [T, B, state_shape]
            a_tm1=a_tm1,  # [T, B]
            r_t=r_t,  # [T, B]
            hidden_s=hidden_s,
        )

    def _put_unroll_onto_queue(self, unrolled_transition):
        # Important note, store hidden states for every step in the unroll will consume HUGE memory.
        self._queue.put(unrolled_transition)

    def _update_actor_network(self):
        state_dict = self._shared_params["network"]

        if state_dict is not None:
            self._network.load_state_dict(state_dict)

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current actor's statistics as a dictionary."""
        return {"exploration_epsilon": self._exploration_epsilon}


class Learner(types_lib.Learner):
    """R2D2 learner"""

    def __init__(
        self,
        network: tf.Module,
        optimizer: keras.optimizers.Optimizer,
        replay: replay_lib.PrioritizedReplay,
        target_net_update_interval: int,
        min_replay_size: int,
        batch_size: int,
        n_step: int,
        discount: float,
        burn_in: int,
        priority_eta: float,
        rescale_epsilon: float,
        clip_grad: bool,
        max_grad_norm: float,
        shared_params: dict,
    ) -> None:
        """
        Args:
            network: the Q network we want to train and optimize.
            optimizer: the optimizer for Q network.
            replay: prioritized recurrent experience replay.
            target_net_update_interval: how often to copy online network parameters to target.
            min_replay_size: wait till experience replay buffer this number before start to learn.
            batch_size: sample batch_size of transitions. [1, 512]
            n_step: TD n-step bootstrap.
            discount: the gamma discount for future rewards. [0.0, 1.0]
            burn_in: burn n transitions to generate initial hidden state before learning.
            priority_eta: coefficient to mix the max and mean absolute TD errors. [0.0, 1.0]
            rescale_epsilon: rescaling factor for n-step targets in the invertible rescaling function. [0.0, 1.0]
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad, only works if clip_grad is True.
            shared_params: a shared dict, so we can later update the parameters for actors.
        """

        self.agent_name = "R2D2-learner"
        self._network = network
        self._optimizer = optimizer
        # Lazy way to create target Q network
        self._target_network = copy.deepcopy(self._network)

        # Disable autograd for target network
        self._target_network.set_trainable(False)

        self._shared_params = shared_params

        self._batch_size = batch_size
        self._n_step = n_step
        self._burn_in = burn_in
        self._target_net_update_interval = target_net_update_interval
        self._discount = discount
        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm
        self._rescale_epsilon = rescale_epsilon

        self._replay = replay
        self._min_replay_size = min_replay_size
        self._priority_eta = priority_eta

        self._max_seen_priority = 1.0  # New unroll will use this as priority

        # Counters
        self._step_t = -1
        self._update_t = 0
        self._target_update_t = 0
        self._loss_t = np.nan

    def step(self) -> Iterable[Mapping[Text, float]]:
        """Increment learner step, and potentially do a update when called.

        Yields:
            learner statistics if network parameters update occurred, otherwise returns None.
        """
        self._step_t += 1

        if (
            self._replay.size < self._min_replay_size
            or self._step_t % max(4, int(self._batch_size * 0.25)) != 0
        ):
            return

        self._learn()
        yield self.statistics

    def reset(self) -> None:
        """Should be called at the beginning of every iteration."""

    def received_item_from_queue(self, item) -> None:
        """Received item send by actors through multiprocessing queue."""
        self._replay.add(item, self._max_seen_priority)

    def get_network_state_dict(self):
        # To keep things consistent, we move the parameters to CPU
        return {k: v.cpu() for k, v in self._network.state_dict().items()}

    def _learn(self) -> None:
        transitions, indices, weights = self._replay.sample(self._batch_size)
        priorities = self._update(transitions, weights)
        self._update_t += 1

        if priorities.shape != (self._batch_size,):
            raise RuntimeError(
                f"Expect priorities has shape ({self._batch_size},), got {priorities.shape}"
            )
        priorities = np.abs(priorities)
        self._max_seen_priority = np.max([self._max_seen_priority, np.max(priorities)])
        self._replay.update_priorities(indices, priorities)

        self._shared_params["network"] = self.get_network_state_dict()

        # Copy online Q network parameters to target Q network, every m updates
        if (
            self._update_t > 1
            and self._update_t % self._target_net_update_interval == 0
        ):
            self._update_target_network()

    def _update(self, transitions: R2d2Transition, weights: np.ndarray) -> np.ndarray:
        """Update online Q network."""
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        assertions_lib.assert_rank_and_dtype(weights, 1, tf.float32)

        # Get initial hidden state, handle possible burn in.
        init_hidden_s = self._extract_first_step_hidden_state(transitions)
        burn_transitions, learn_transitions = replay_lib.split_structure(
            transitions, TransitionStructure, self._burn_in
        )
        if burn_transitions is not None:
            hidden_s, target_hidden_s = self._burn_in_unroll_q_networks(
                burn_transitions, init_hidden_s
            )
        else:
            hidden_s = tuple(tf.identity(s) for s in init_hidden_s)
            target_hidden_s = tuple(tf.identity(s) for s in init_hidden_s)

        with tf.GradientTape() as tape:
            loss, priorities = self._calc_loss(
                learn_transitions, hidden_s, target_hidden_s
            )

            # Multiply loss by sampling weights, averaging over batch dimension
            loss = tf.math.reduce_mean(loss * tf.stop_gradient(weights))

        tf.debugging.check_numerics(loss, "Loss has NaNs or Inf.")
        grads = tape.gradient(loss, self._network.trainable_variables)
        if self._clip_grad:
            grads, _ = tf.clip_by_global_norm(grads, self._max_grad_norm)
        self._optimizer.apply_gradients(zip(grads, self._network.trainable_variables))

        # For logging only.
        self._loss_t = loss.numpy()
        return priorities.numpy()

    def _calc_loss(
        self,
        transitions: R2d2Transition,
        hidden_s: HiddenState,
        target_hidden_s: HiddenState,
    ) -> Tuple[tf.Tensor, np.ndarray]:
        """Calculate loss and priorities for given unroll sequence transitions."""
        s_t = tf.convert_to_tensor(transitions.s_t, dtype=tf.float32)
        a_t = tf.convert_to_tensor(transitions.a_t, dtype=tf.int64)
        last_action = tf.convert_to_tensor(transitions.last_action, dtype=tf.int64)
        r_t = tf.convert_to_tensor(transitions.r_t, dtype=tf.float32)
        done = tf.convert_to_tensor(transitions.done, dtype=tf.bool)

        # Rank and dtype checks, note we have a new unroll time dimension, states may be images, which is rank 5.
        assertions_lib.assert_rank_and_dtype(s_t, (3, 5), tf.float32)
        assertions_lib.assert_rank_and_dtype(a_t, 2, tf.int64)
        assertions_lib.assert_rank_and_dtype(last_action, 2, tf.int64)
        assertions_lib.assert_rank_and_dtype(r_t, 2, tf.float32)
        assertions_lib.assert_rank_and_dtype(done, 2, tf.bool)

        # Get q values from online Q network
        q_t = self._network(
            RnnDqnNetworkInputs(s_t=s_t, a_tm1=last_action, r_t=r_t, hidden_s=hidden_s)
        ).q_values

        # Computes raw target q values, use double Q
        best_a_t = tf.math.argmax(q_t, axis=-1)  # [T, B]

        target_q_t = self._target_network(
            RnnDqnNetworkInputs(
                s_t=s_t, a_tm1=last_action, r_t=r_t, hidden_s=target_hidden_s
            )
        ).q_values

        # Calculate losses and priorities using TensorFlow operations
        losses, priorities = calculate_losses_and_priorities(
            q_value=q_t,
            action=a_t,
            reward=r_t,
            done=done,
            target_q_value=target_q_t,
            target_action=best_a_t,
            gamma=self._discount,
            n_step=self._n_step,
            eps=self._rescale_epsilon,
            eta=self._priority_eta,
        )

        return (losses, priorities)

    def _burn_in_unroll_q_networks(
        self, transitions: R2d2Transition, init_hidden_s: HiddenState
    ) -> Tuple[HiddenState, HiddenState]:
        """Unroll both online and target q networks to generate hidden states for LSTM."""
        s_t = tf.convert_to_tensor(
            transitions.s_t, dtype=tf.float32
        )  # [burn_in, B, state_shape]
        last_action = tf.convert_to_tensor(
            transitions.last_action, dtype=tf.int64
        )  # [burn_in, B]
        r_t = tf.convert_to_tensor(transitions.r_t, dtype=tf.float32)  # [burn_in, B]

        # Rank and dtype checks, note we have a new unroll time dimension, states may be images, which is rank 5.
        assertions_lib.assert_rank_and_dtype(s_t, (3, 5), tf.float32)
        assertions_lib.assert_rank_and_dtype(last_action, 2, tf.int64)
        assertions_lib.assert_rank_and_dtype(r_t, 2, tf.float32)

        _hidden_s = tuple(tf.identity(s) for s in init_hidden_s)
        _target_hidden_s = tuple(tf.identity(s) for s in init_hidden_s)

        # Burn in to generate hidden states for LSTM, we unroll both online and target Q networks
        hidden_s = self._network(
            RnnDqnNetworkInputs(s_t=s_t, a_tm1=last_action, r_t=r_t, hidden_s=_hidden_s)
        ).hidden_s
        target_hidden_s = self._target_network(
            RnnDqnNetworkInputs(
                s_t=s_t, a_tm1=last_action, r_t=r_t, hidden_s=_target_hidden_s
            )
        ).hidden_s

        return (hidden_s, target_hidden_s)

    def _extract_first_step_hidden_state(
        self, transitions: R2d2Transition
    ) -> HiddenState:
        # We only need the first step hidden states in replay, shape [batch_size, num_lstm_layers, lstm_hidden_size]
        init_h = tf.convert_to_tensor(transitions.init_h[:1], dtype=tf.float32)
        init_c = tf.convert_to_tensor(transitions.init_c[:1], dtype=tf.float32)

        # Rank and dtype checks.
        assertions_lib.assert_rank_and_dtype(init_h, 3, tf.float32)
        assertions_lib.assert_rank_and_dtype(init_c, 3, tf.float32)

        # Swap batch and num_lstm_layers axis.
        init_h = tf.transpose(init_h, perm=[1, 0, 2])
        init_c = tf.transpose(init_c, perm=[1, 0, 2])

        # Batch dimension checks.
        assertions_lib.assert_batch_dimension(init_h, self._batch_size, 1)
        assertions_lib.assert_batch_dimension(init_c, self._batch_size, 1)

        return (init_h, init_c)

    def _update_target_network(self):
        self._target_network.set_weights(self._network.get_weights())
        self._target_update_t += 1

    @property
    def statistics(self) -> Mapping[str, float]:
        """Returns current agent statistics as a dictionary."""
        return {
            # 'learning_rate': self._optimizer.param_groups[0]['lr'],
            "loss": self._loss_t,
            # 'discount': self._discount,
            "updates": self._update_t,
            "target_updates": self._target_update_t,
        }


class R2d2DqnMlpNet(tf.Module):
    """R2D2 DQN MLP network."""

    def __init__(self, state_dim: int, action_dim: int):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network
            action_dim: the number of units for the output liner layer
        """

        super().__init__()
        self.action_dim = action_dim

        self.body = keras.Sequential(
            [
                keras.layers.Dense(64),
                keras.layers.ReLU(),
                keras.layers.Dense(128),
                keras.layers.ReLU(),
            ]
        )

        self.lstm_hidden_size = 128
        self.lstm = keras.layers.LSTM(self.lstm_hidden_size, return_sequences=False)

        self.advantage_head = keras.Sequential(
            [
                keras.layers.Dense(128),
                keras.layers.ReLU(),
                keras.layers.Dense(action_dim),
            ]
        )
        self.value_head = keras.Sequential(
            [
                keras.layers.Dense(128),
                keras.layers.ReLU(),
                keras.layers.Dense(1),
            ]
        )

    def forward(self, input_: RnnDqnNetworkInputs) -> RnnDqnNetworkOutputs:
        # Expect x shape to be [T, B, state_shape]
        s_t = input_.s_t
        a_tm1 = input_.a_tm1
        r_t = input_.r_t
        hidden_s = input_.hidden_s

        T, B, *_ = s_t.shape  # [T, B, state_shape]
        x = tf.reshape(s_t, (1,))  # Merge batch and time dimension.

        x = self.body(x)
        x = x.view(T * B, -1)

        # Append reward and one hot last action.
        one_hot_a_tm1 = tf.one_hot(a_tm1.view(T * B), self.action_dim).float()

        reward = r_t.view(T * B, 1)
        core_input = tf.concat([x, reward, one_hot_a_tm1], axis=0)
        core_input = core_input.view(T, B, -1)  # LSTM expect rank 3 tensor.

        # If no hidden_s provided, use zero start strategy
        if hidden_s is None:
            hidden_s = self.get_initial_hidden_state(batch_size=B)
            hidden_s = tuple(hidden_s)

        x, hidden_s = self.lstm(core_input, hidden_s)

        x = tf.reshape(x, (1,))  # Merge batch and time dimension.
        advantages = self.advantage_head(x)  # [T*B, action_dim]
        values = self.value_head(x)  # [T*B, 1]

        q_values = values + (
            advantages - tf.math.reduce_mean(advantages, axis=1, keepdims=True)
        )
        q_values = q_values.view(T, B, -1)  # reshape to in the range [B, T, action_dim]
        return RnnDqnNetworkOutputs(q_values=q_values, hidden_s=hidden_s)

    def get_initial_hidden_state(self, batch_size: int) -> Tuple[tf.Tensor]:
        """Get initial LSTM hidden state, which is all zeros,
        should call at the beginning of new episode, or every training batch"""
        # Shape should be num_layers, batch_size, hidden_size, note lstm expects two hidden states.
        return tuple(
            tf.zeros((1, batch_size, self.lstm_hidden_size), dtype=tf.float32)
            for _ in range(2)
        )

    def set_trainable(self, trainable: bool):
        for net in [self.body, self.advantage_head, self.value_head]:
            net.trainable = trainable
