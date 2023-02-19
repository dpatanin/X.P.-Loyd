import random
import numpy as np
from collections import deque


class ExperienceReplayBuffer:
    def __init__(self, max_size: int, goal_sampling_strategy="final"):
        self.buffer = deque(maxlen=max_size)
        self.goal_sampling_strategy = goal_sampling_strategy

    def add(self, experience: tuple[np.ndarray, int, float, np.ndarray, bool]):
        self.buffer.append(experience)

    def recall(self, reach: int):
        """
        Recalls the latest experiences from memory.
        Returns: [states, actions, rewards, next_states, dones]
        """
        if len(self.buffer) < reach:
            reach = len(self.buffer)
        experiences = list(self.buffer)[-reach:]
        return tuple(np.array(e) for e in zip(*experiences))

    def sample(self, batch_size: int):
        """
        Samples random experiences from memory.
        Returns: [states, actions, rewards, next_states, dones]
        """
        experiences = random.sample(self.buffer, batch_size)
        return tuple(np.array(e) for e in zip(*experiences))

    def __len__(self):
        return len(self.buffer)
