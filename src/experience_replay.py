import random
import numpy as np
from collections import deque


class ExperienceReplayBuffer:
    def __init__(self, max_size: int, goal_sampling_strategy="final"):
        self.buffer = deque(maxlen=max_size)
        self.goal_sampling_strategy = goal_sampling_strategy

    def add(self, experience: tuple[np.ndarray, int, float, np.ndarray, bool]):
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        """
        Returns: [states, actions, rewards, next_states, dones]
        """
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = (
            [],
            [],
            [],
            [],
            [],
        )

        for experience in experiences:
            (state, action, reward, next_state, done) = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)
