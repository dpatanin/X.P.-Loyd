import collections
import copy
import itertools
from typing import Any, Iterable, NamedTuple, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np

CompressedArray = Tuple[bytes, Tuple, np.dtype]

# Generic replay structure: Any flat named tuple.
ReplayStructure = TypeVar("ReplayStructure", bound=Tuple[Any, ...])


class Transition(NamedTuple):
    """A full transition for general use case"""

    s_tm1: Optional[np.ndarray]
    a_tm1: Optional[int]
    r_t: Optional[float]
    s_t: Optional[np.ndarray]
    done: Optional[bool]


TransitionStructure = Transition(s_tm1=None, a_tm1=None, r_t=None, s_t=None, done=None)


class PrioritizedReplay:
    """Prioritized replay, with circular buffer storage for flat named tuples.
    This is the proportional variant as described in
    http://arxiv.org/abs/1511.05952.

    """

    def __init__(
        self,
        capacity: int,
        structure: ReplayStructure,
        priority_exponent: float,
        importance_sampling_exponent: float,
        random_state: np.random.RandomState,
        normalize_weights: bool = True,
        time_major: bool = False,
    ):
        if capacity <= 0:
            raise ValueError(
                f"Expect capacity to be a positive integer, got {capacity}"
            )
        self.structure = structure
        self._capacity = capacity
        self._random_state = random_state

        self._storage = [None] * capacity
        self._num_added = 0

        self._time_major = time_major

        self._priorities = np.ones((capacity,), dtype=np.float32)
        self._priority_exponent = priority_exponent
        self._importance_sampling_exponent = importance_sampling_exponent

        self._normalize_weights = normalize_weights

    def add(self, item: ReplayStructure, priority: float) -> None:
        """Adds a single item with a given priority to the replay buffer."""
        if not np.isfinite(priority) or priority < 0.0:
            raise ValueError("priority must be finite and positive.")

        index = self._num_added % self._capacity
        self._priorities[index] = priority
        self._storage[index] = item
        self._num_added += 1

    def get(self, indices: Sequence[int]) -> Iterable[ReplayStructure]:
        """Retrieves transitions by indices."""
        return [self._storage[i] for i in indices]

    def sample(self, size: int) -> Tuple[ReplayStructure, np.ndarray, np.ndarray]:
        """Samples a batch of transitions."""
        if self.size < size:
            raise RuntimeError(
                f"Replay only have {self.size} samples, got sample size {size}"
            )

        if self._priority_exponent == 0:
            indices = self._random_state.uniform(0, self.size, size=size).astype(
                np.int64
            )
            weights = np.ones_like(indices, dtype=np.float32)
        else:
            # code copied from seed_rl
            priorities = self._priorities[: self.size] ** self._priority_exponent

            probs = priorities / np.sum(priorities)
            indices = self._random_state.choice(
                np.arange(probs.shape[0]), size=size, replace=True, p=probs
            )

            # Importance weights.
            weights = (
                (1.0 / self.size) / np.take(probs, indices)
            ) ** self.importance_sampling_exponent

            if self._normalize_weights:
                weights /= np.max(weights) + 1e-8  # Normalize.

        samples = self.get(indices)
        stacked = np_stack_list_of_transitions(samples, self.structure, self.stack_dim)
        return stacked, indices, weights

    def update_priorities(
        self, indices: Sequence[int], priorities: Sequence[float]
    ) -> None:
        """Updates indices with given priorities."""
        priorities = np.asarray(priorities)
        if not np.isfinite(priorities).all() or (priorities < 0.0).any():
            raise ValueError("priorities must be finite and positive.")
        for index, priority in zip(indices, priorities):
            self._priorities[index] = priority

    def stack_dim(self) -> int:
        """Stack dimension, for RNN we may need to make the tensor time major by stacking on second dimension as [T, B, ...]."""
        return 1 if self._time_major else 0

    @property
    def size(self) -> None:
        """Number of elements currently contained in replay."""
        return min(self._num_added, self._capacity)

    @property
    def capacity(self) -> None:
        """Total capacity of replay (maximum number of items that can be stored)."""
        return self._capacity

    @property
    def importance_sampling_exponent(self):
        """Importance sampling exponent at current step."""
        return self._importance_sampling_exponent(self._num_added)


class Unroll:
    """Unroll transitions to a specific timestep, used for RNN networks like R2D2, IMPALA,
    support cross episodes and do not cross episodes."""

    def __init__(
        self,
        unroll_length: int,
        overlap: int,
        structure: ReplayStructure,
        cross_episode: bool = True,
    ) -> None:
        """
        Args:
            unroll_length: the unroll length.
            overlap: adjacent unrolls overlap.
            structure: transition structure, used to stack sequence of unrolls into a single transition.
            cross_episode: should unroll cross episode, default on.
        """

        self.structure = structure

        self._unroll_length = unroll_length
        self._overlap = overlap
        self._full_unroll_length = unroll_length + overlap
        self._cross_episode = cross_episode

        self._storage = collections.deque(maxlen=self._full_unroll_length)

        # Persist last unrolled transitions incase not cross episode.
        # Sometimes the episode ends without reaching a full 'unroll length',
        # we will reuse some transitions from last unroll to try to make a 'full length unroll'.
        self._last_unroll = None

    def add(self, transition: Any, done: bool) -> Union[ReplayStructure, None]:
        """Add new transition into storage."""
        self._storage.append(transition)

        if self.full:
            return self._pack_unroll_into_single_transition()
        return self._handle_episode_end() if done else None

    def _pack_unroll_into_single_transition(self) -> Union[ReplayStructure, None]:
        """Return a single transition object with transitions stacked with the unroll structure."""
        if not self.full:
            return None

        _sequence = list(self._storage)
        # Save for later use.
        self._last_unroll = copy.deepcopy(_sequence)
        self._storage.clear()

        # Handling adjacent unroll sequences overlapping
        if self._overlap > 0:
            for transition in _sequence[-self._overlap :]:  # noqa: E203
                self._storage.append(transition)
        return self._stack_unroll(_sequence)

    def _handle_episode_end(self) -> Union[ReplayStructure, None]:
        """Handle episode end, incase no cross episodes, try to build a full unroll if last unroll is available."""
        if self._cross_episode:
            return None
        if self.size <= 0 or self._last_unroll is None:
            return None
        # Incase episode ends without reaching a full 'unroll length'
        # Use whatever we got from current unroll, fill in the missing ones from previous sequence
        _suffix = list(self._storage)
        _prefix_indices = self._full_unroll_length - len(_suffix)
        _prefix = self._last_unroll[-_prefix_indices:]
        _sequence = list(itertools.chain(_prefix, _suffix))
        return self._stack_unroll(_sequence)

    def reset(self):
        """Reset unroll storage."""
        self._storage.clear()
        self._last_unroll = None

    def _stack_unroll(self, sequence):
        if len(sequence) != self._full_unroll_length:
            raise RuntimeError(
                f"Expect sequence length to be {self._full_unroll_length}, got {len(sequence)}"
            )
        return np_stack_list_of_transitions(sequence, self.structure)

    @property
    def size(self):
        """Return current unroll size."""
        return len(self._storage)

    @property
    def full(self):
        """Return is unroll full."""
        return len(self._storage) == self._storage.maxlen


def np_stack_list_of_transitions(transitions, structure, axis=0):
    """
    Stack list of transition objects into one transition object with lists of tensors
    on a given dimension (default 0)
    """

    transposed = zip(*transitions)
    stacked = [np.stack(xs, axis=axis) for xs in transposed]
    return type(structure)(*stacked)


def split_structure(
    input_, structure, prefix_length: int, axis: int = 0
) -> Tuple[ReplayStructure]:
    """Splits a structure of np.array along the axis, default 0."""

    # Compatibility check.
    if prefix_length > 0:
        for v in input_:
            if v.shape[axis] < prefix_length:
                raise ValueError(
                    f"Expect prefix_length to be less or equal to {v.shape[axis]}, got {prefix_length}"
                )

    if prefix_length == 0:
        return (None, input_)
    split = [
        np.split(
            xs,
            [
                prefix_length,
                xs.shape[axis],
            ],  # for torch.split() [prefix_length, xs.shape[axis] - prefix_length],
            axis=axis,
        )
        for xs in input_
    ]

    _prefix = [pair[0] for pair in split]
    _suffix = [pair[1] for pair in split]

    return (type(structure)(*_prefix), type(structure)(*_suffix))
