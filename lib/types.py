import abc
from typing import Any, Iterable, Mapping, NamedTuple, Optional, Text

import numpy as np

from lib.trading_env import TradingStats

Action = int


class TimeStep(NamedTuple):
    """Environment timestep"""

    observation: Optional[np.ndarray]
    reward: Optional[float]  # reward from the environment, could be clipped or scaled
    done: Optional[
        bool
    ]  # termination mark of a episode, could also be loss-of-life for Atari
    first: Optional[bool]  # first step of an episode
    info: Optional[
        TradingStats
    ]  # Info dictionary which contains non-clipped/unscaled reward and other information, only used by the trackers


class Agent(abc.ABC):
    """Agent interface."""

    agent_name: str  # agent name
    step_t: int  # runtime steps

    @abc.abstractmethod
    def step(self, timestep: TimeStep) -> Action:
        """Selects action given timestep and potentially learns."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Resets the agent's episodic state such as frame stack and action repeat.

        This method should be called at the beginning of every episode.
        """

    @property
    @abc.abstractmethod
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""


class Learner(abc.ABC):
    """Learner interface."""

    agent_name: str  # agent name
    step_t: int  # learner steps

    @abc.abstractmethod
    def step(self) -> Iterable[Mapping[Text, float]]:
        """Increment learner step, and potentially do a update when called.

        Yields:
            learner statistics if network parameters update occurred, otherwise None.
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """Should be called at the beginning of every iteration."""

    @abc.abstractmethod
    def received_item_from_queue(self, item: Any) -> None:
        """Received item send by actors through multiprocessing queue."""

    @property
    @abc.abstractmethod
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
