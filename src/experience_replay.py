import random
from src.state import State
from collections import deque
from typing import Deque, Tuple


class ExperienceReplayBuffer:
    def __init__(self, max_size: int):
        self.buffer: Deque[Tuple["State", float, float, "State", bool]] = deque(
            maxlen=max_size
        )

    def add(self, experience: Tuple["State", float, float, "State", bool]) -> None:
        self.buffer.append(experience)

    def sample(
        self, batch_size: int
    ) -> Tuple[list["State"], list[float], list[float], list["State"], list[bool]]:
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough experiences in the buffer.")

        experiences = random.sample(self.buffer, batch_size)
        states, predictions, rewards, next_states, dones = zip(*experiences)

        return (
            list(states),
            list(predictions),
            list(rewards),
            list(next_states),
            list(dones),
        )

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


class HERBuffer(ExperienceReplayBuffer):
    def __init__(self, max_size: int, reward_factor=1):
        super().__init__(max_size)
        self.reward_factor = reward_factor

    def analyze_missed_opportunities(self, price_per_contract):
        experiences = self.remember_last_episode()

        # alt state represents current state per iteration during alt timeline
        alt_state: State = None
        price_shift_ref = 0.00
        for xp in experiences:
            s, q, r, ns, d = xp

            # TODO: reward intrinsic motivation
            current_price: float = s.data["Close"].iloc[-1]
            price_diff: float = ns.data["Close"].iloc[-1] - current_price

            if (
                self.__check_missed_opportunity(s.contracts, ns.contracts, price_diff)
                and not alt_state
            ):
                price_shift_ref = price_diff

                if price_diff > 0:
                    ns.enter_long(current_price, 1, price_per_contract)
                    self.add((s, 1, r, ns, d))
                if price_diff < 0:
                    ns.enter_short(current_price, 1, price_per_contract)
                    self.add((s, 2, r, ns, d))
                alt_state = ns

            elif alt_state:
                if (
                    self.__check_price_shift(price_shift_ref, price_diff)
                    or ns.contracts != 0
                ):
                    r = self.reward_factor * alt_state.exit_position(
                        current_price, price_per_contract
                    )
                    self.add((alt_state, 3, r, ns, d))
                    alt_state = None
                else:
                    ns.balance = alt_state.balance
                    ns.contracts = alt_state.contracts
                    ns.entry_price = alt_state.entry_price
                    self.add((alt_state, q, r, ns, d))
                    alt_state = ns

    def remember_last_episode(self) -> list[Tuple[State, int, float, State, bool]]:
        experiences = list(self.buffer)
        experiences.reverse()
        last_episode = []

        # Latest experience will have done == True, thus counter looking for the second done
        dones = 0
        for xp in experiences:
            if xp[-1]:
                dones += 1

            if dones < 2:
                last_episode.append(xp)

        last_episode.reverse()
        return last_episode

    def __check_price_shift(self, ref: float, diff: float):
        return (ref < 0 and diff > 0) or (ref > 0 and diff < 0)

    def __check_missed_opportunity(self, c1: int, c2: int, diff: float):
        return c1 == 0 and c2 == 0 and diff != 0
