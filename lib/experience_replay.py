import random
from collections import deque
from typing import Deque, Tuple

from lib.action_space import ActionSpace
from lib.constants import CLOSE
from lib.state import State


class Memory:
    def __init__(
        self,
        origin: State = None,
        q_value: float = None,
        reward: float = None,
        outcome: State = None,
        done: bool = None,
    ):
        self.origin = origin
        self.q_value = q_value
        self.reward = reward
        self.outcome = outcome
        self.done = done

    def is_complete(self):
        return (
            self.origin is not None
            and self.q_value is not None
            and self.reward is not None
            and self.outcome is not None
            and self.done is not None
        )

    def copy(self):
        return Memory(
            origin=self.origin,
            q_value=self.q_value,
            reward=self.reward,
            outcome=self.outcome,
            done=self.done,
        )


class ExperienceReplayBuffer:
    """
    A basic experience replay buffer representing a collection of transitions.\n
    Unlike in a standard experience replay, this does not store predictions
    as we use one continuous value and derive the actions therefrom.

    One experience contains: `[State used for prediction, Reward after action, Next state after action, Flag for session end]`
    |`max_size`: Maximum amount of experiences being stored. (New delete oldest when full.)
    """

    def __init__(self, max_size=2000):
        self.buffer: Deque[Memory] = deque(maxlen=max_size)

    def add(self, experience: Memory) -> None:
        assert experience.is_complete(), "Received incomplete experience."
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Memory]:
        """
        Randomly samples `batch_size` transitions/experiences.
        """

        if len(self.buffer) < batch_size:
            raise ValueError("Not enough experiences in the buffer.")

        return random.sample(self.buffer, batch_size)

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


class HERBuffer(ExperienceReplayBuffer):
    """
    A extended experience replay using the principle of a hindsight replay.\n
    It creates alternative virtual experiences along the real ones.
    """

    def __init__(self, max_size=2000):
        super().__init__(max_size)

    def add(self, experience: Memory) -> None:
        super().add(experience)

        alt_xp = experience.copy()
        alt_xp.q_value *= -1
        alt_xp.reward *= -1
        super().add(alt_xp)
