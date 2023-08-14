import numpy as np
import tensorflow as tf

import lib.assertions as assertions_lib


def calculate_dist_priorities_from_td_error(
    td_errors: tf.Tensor, eta: float
) -> np.ndarray:
    """Calculate priorities for distributed experience replay, follows Ape-x and R2D2 papers."""

    td_errors = tf.identity(td_errors)  # Create a copy to detach
    abs_td_errors = tf.abs(td_errors)

    max_abs_td_errors = tf.math.reduce_max(abs_td_errors, axis=0)
    mean_abs_td_errors = tf.math.reduce_mean(abs_td_errors, axis=0)

    priorities = eta * max_abs_td_errors + (1 - eta) * mean_abs_td_errors
    priorities = tf.clip_by_value(
        priorities, clip_value_min=0.0001, clip_value_max=1000
    )  # Avoid NaNs
    priorities = priorities.numpy()

    return priorities


def signed_hyperbolic(x: tf.Tensor, eps: float = 1e-3) -> tf.Tensor:
    """Signed hyperbolic transform, inverse of signed_parabolic."""
    assertions_lib.assert_dtype(
        x, (tf.dtypes.float16, tf.dtypes.float32, tf.dtypes.float64)
    )
    return tf.sign(x) * (tf.sqrt(tf.abs(x) + 1) - 1) + eps * x


def signed_parabolic(x: tf.Tensor, eps: float = 1e-3) -> tf.Tensor:
    """Signed parabolic transform, inverse of signed_hyperbolic."""
    assertions_lib.assert_dtype(
        x, (tf.dtypes.float16, tf.dtypes.float32, tf.dtypes.float64)
    )
    z = tf.sqrt(1 + 4 * eps * (eps + 1 + tf.abs(x))) / 2 / eps - 1 / 2 / eps
    return tf.sign(x) * (tf.square(z) - 1)


def n_step_bellman_target(
    r_t: tf.Tensor,
    done: tf.Tensor,
    q_t: tf.Tensor,
    gamma: float,
    n_steps: int,
) -> tf.Tensor:
    r"""Computes n-step Bellman targets.

    See section 2.3 of R2D2 paper (which does not mention the logic around end of
    episode).

    Args:
      rewards: This is r_t in the equations below. Should be non-discounted, non-summed,
        shape [T, B] tensor.
      done: This is done_t in the equations below. done_t should be true
        if the episode is done just after
        experimenting reward r_t, shape [T, B] tensor.
      q_t: This is Q_target(s_{t+1}, a*) (where a* is an action chosen by the caller),
        shape [T, B] tensor.
      gamma: Exponential RL discounting.
      n_steps: The number of steps to look ahead for computing the Bellman targets.

    Returns:
      y_t targets as <float32>[time, batch_size] tensor.
      When n_steps=1, this is just:

      $$r_t + gamma * (1 - done_t) * Q_{target}(s_{t+1}, a^*)$$

      In the general case, this is:

      $$(\sum_{i=0}^{n-1} \gamma ^ {i} * notdone_{t, i-1} * r_{t + i}) +
        \gamma ^ n * notdone_{t, n-1} * Q_{target}(s_{t + n}, a^*) $$

      where notdone_{t,i} is defined as:

      $$notdone_{t,i} = \prod_{k=0}^{k=i}(1 - done_{t+k})$$

      The last n_step-1 targets cannot be computed with n_step returns, since we
      run out of Q_{target}(s_{t+n}). Instead, they will use n_steps-1, .., 1 step
      returns. For those last targets, the last Q_{target}(s_{t}, a^*) is re-used
      multiple times.
    """
    # Rank and compatibility checks.
    assertions_lib.assert_rank_and_dtype(r_t, 2, tf.dtypes.float32)
    assertions_lib.assert_rank_and_dtype(done, 2, tf.dtypes.bool)
    assertions_lib.assert_rank_and_dtype(q_t, 2, tf.dtypes.float32)

    assertions_lib.assert_batch_dimension(done, q_t.shape[0])
    assertions_lib.assert_batch_dimension(r_t, q_t.shape[0])
    assertions_lib.assert_batch_dimension(done, q_t.shape[1], 1)
    assertions_lib.assert_batch_dimension(r_t, q_t.shape[1], 1)

    # We append n_steps - 1 times the last q_target. They are divided by gamma **
    # k to correct for the fact that they are at a 'fake' indices, and will
    # therefore end up being multiplied back by gamma ** k in the loop below.
    # We prepend 0s that will be discarded at the first iteration below.
    bellman_target = tf.concat(
        [tf.zeros_like(q_t[:1]), q_t]
        + [q_t[-1:] / gamma**k for k in range(1, n_steps)],
        dim=0,
    )
    # Pad with n_steps 0s. They will be used to compute the last n_steps-1
    # targets (having 0 values is important).
    done = tf.concat([done] + [tf.zeros_like(done[:1])] * n_steps, dim=0)
    rewards = tf.concat([r_t] + [tf.zeros_like(r_t[:1])] * n_steps, dim=0)
    # Iteratively build the n_steps targets. After the i-th iteration (1-based),
    # bellman_target is effectively the i-step returns.
    for _ in range(n_steps):
        rewards = rewards[:-1]
        done = done[:-1]
        bellman_target = rewards + gamma * (1.0 - done.float()) * bellman_target[1:]

    return bellman_target
