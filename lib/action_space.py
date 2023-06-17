from lib.state import State


class ActionSpace:
    """
    Responsible for taking the action, this class defines the trading strategy and returns the respective rewards.

    |`threshold`: The threshold of the prediction/value to act. (0 - 1); 0: Always act; 1: Never act.
    |`price_per_contract`: The price of a single contract (or item in general).
    |`limit`: Absolute trading limit per single trade.
    |`intrinsic_fac`: Weight for intrinsic rewards.

    Strategy:\n
    If under threshold, do nothing unless prediction opposes current position, in that case be careful and exit position.
    If above threshold, enter the predicted position. If already in that position, keep your contracts and reenter.
    """

    def __init__(
        self, threshold: float, price_per_contract: float, limit: int, intrinsic_fac=1
    ):
        self.threshold = threshold
        self.ppc = price_per_contract
        self.limit = limit
        self.intrinsic_fac = intrinsic_fac

    def calc_trade_amount(self, q: float, state: "State") -> int:
        """
        Scales the q value prediction to the amount of contracts to trade.
        """
        max_amount = min(state.data["volume"].median(), self.limit)
        return round(
            abs(((abs(q) - self.threshold) / (1 - self.threshold)) * max_amount)
        )

    def remove_overhead(self, contracts: int, balance: float) -> int:
        """
        Calculates the overhead of contracts in regard to the balance.
        Returns the new calculated trade amount without the overhead.
        """
        if balance <= 0:
            return 0

        overhead = balance - (contracts * self.ppc)
        return (
            contracts - round(abs(overhead / self.ppc)) if overhead <= 0 else contracts
        )

    def take_action(self, q: float, state: "State") -> float:
        if q == 0:
            return 0

        abs_q = abs(q)
        position = state.has_position()
        amount = self.calc_trade_amount(q, state)
        reward = 0

        if abs_q < self.threshold:
            if self.is_opposite_direction(q, position):
                reward = state.exit_position(self.ppc)
        else:
            if position:
                if not self.is_opposite_direction(q, position):
                    amount += abs(state.contracts)
                reward = state.exit_position(self.ppc)

            if q > 0:
                amount = self.remove_overhead(amount, state.balance)
                state.enter_long(amount, self.ppc)
            else:
                state.enter_short(amount, self.ppc)

            reward += amount * self.ppc * self.intrinsic_fac

        return reward

    def is_opposite_direction(self, q: float, position: int) -> bool:
        return (q > 0 and position < 0) or (q < 0 and position > 0)
