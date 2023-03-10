from src.state import State
import numpy as np

# TODO: better doc
class ActionSpace:
    """
    Responsible for taking the action, this class defines the trading strategy and part of the rewards.

    Strategy:\n
    If under threshold, do nothing unless direction changes, in that case be careful and exit position.
    If above threshold, enter the desired direction. If already in that direction, keep your contracts (reenter).
    """

    def __init__(self, threshold: float, price_per_contract: float, limit: int):
        self.threshold = threshold
        self.ppc = price_per_contract
        self.limit = limit

    def calc_trade_amount(self, q: float, state: "State") -> int:
        max_amount = min(state.data["Volume"].median(), self.limit)
        return round(
            abs(((abs(q) - self.threshold) / (1 - self.threshold)) * max_amount)
        )

    def calc_overhead(self, contracts: int, balance: float) -> int:
        return round(
            overhead / self.ppc
            if (overhead := balance - (contracts * self.ppc)) < 0
            else 0
        )

    def take_action(self, q: float, state: "State") -> float:
        if q == 0:
            return 0

        abs_q = abs(q)
        position = state.has_position()
        price = state.data["Close"].iloc[-1]
        amount = self.calc_trade_amount(q, state)
        reward = 0

        if abs_q < self.threshold:
            if self.is_opposite_direction(q, position):
                reward = state.exit_position(price, self.ppc)
        else:
            if self.is_opposite_direction(q, position):
                reward = state.exit_position(price, self.ppc)
            elif position:
                amount += abs(state.contracts)
                state.exit_position(price, self.ppc)

            state.enter_long(
                price, amount - self.calc_overhead(amount, state.balance), self.ppc
            ) if q > 0 else state.enter_short(price, amount, self.ppc)

        return reward

    def is_opposite_direction(self, q: float, position: int) -> bool:
        return (q > 0 and position < 0) or (q < 0 and position > 0)
