import random
import numpy as np
from collections import deque
from typing import Deque, Tuple


class ExperienceReplayBuffer:
    def __init__(self, max_size: int, goal_sampling_strategy: str = "final"):
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=max_size
        )
        self.goal_sampling_strategy: str = goal_sampling_strategy

    def add(self, experience: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        assert all(
            e[0].shape == experience[0].shape for e in self.buffer
        ), "State shape mismatch."
        assert all(
            e[3].shape == experience[3].shape for e in self.buffer
        ), "Next state shape mismatch."

        self.buffer.append(experience)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough experiences in the buffer.")

        experiences = np.random.choice(self.buffer, size=batch_size, replace=False)
        return tuple(np.array(e) for e in zip(*experiences))

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)
