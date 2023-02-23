import random
import numpy as np
from src.state import State
from collections import deque
from typing import Deque, Tuple


class ExperienceReplayBuffer:
    def __init__(self, max_size: int, goal_sampling_strategy: str = "final"):
        self.buffer: Deque[Tuple["State", int, float, "State", bool]] = deque(
            maxlen=max_size
        )
        self.goal_sampling_strategy: str = goal_sampling_strategy

    def add(self, experience: Tuple["State", int, float, "State", bool]) -> None:
        self.buffer.append(experience)

    def sample(
        self, batch_size: int
    ) -> Tuple[list["State"], list[int], list[float], list["State"], list[bool]]:
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough experiences in the buffer.")

        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        return (
            list(states),
            list(actions),
            list(rewards),
            list(next_states),
            list(dones),
        )

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)
