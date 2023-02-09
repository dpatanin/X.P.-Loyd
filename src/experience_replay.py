import random
import numpy as np
from collections import deque

class HindsightExperienceReplay:
    def __init__(self, max_size: int, goal_sampling_strategy="final"):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        self.goal_sampling_strategy = goal_sampling_strategy

    def add(self, experience: tuple[np.ndarray, int, float, np.ndarray, bool]):
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        """
        Returns: [states, actions, rewards, next_states, dones, achieved_goals, desired_goals]
        """
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        achieved_goals, desired_goals = [], []

        for experience in experiences:
            state, action, reward, next_state, done = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            if self.goal_sampling_strategy == "final":
                desired_goal = next_state[-3:]
            elif self.goal_sampling_strategy == "future":
                desired_goal = np.random.uniform(-1, 1, size=3)
            else:
                raise ValueError("Invalid goal sampling strategy.")

            achieved_goal = state[-3:]
            desired_goals.append(desired_goal)
            achieved_goals.append(achieved_goal)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(achieved_goals),
            np.array(desired_goals),
        )

    def __len__(self):
        return len(self.buffer)
