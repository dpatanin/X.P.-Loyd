from lib.state import State
from lib.constants import VOLUME, ACTION_SHORT, ACTION_LONG, ACTION_EXIT, ACTION_STAY


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

    def __init__(self, threshold: float, limit: int, intrinsic_fac=1):
        self.threshold = threshold
        self.limit = limit
        self.intrinsic_fac = intrinsic_fac

    def calc_trade_amount(self, q: float, state: "State") -> int:
        """
        Scales the q value prediction to the amount of contracts to trade.
        Returns at least the amount of `1` contracts.
        """
        max_amount = min(state.data[VOLUME].median(), self.limit)
        return max(
            round(abs(((abs(q) - self.threshold) / (1 - self.threshold)) * max_amount)),
            1,
        )

    def take_action(self, q: float, state: "State"):
        """
        Takes action on a state inplace.
        Returns tuple with reward and taken action.
        """
        action = ACTION_STAY
        reward = 0.00
        abs_q = abs(q)
        amount = self.calc_trade_amount(q, state)

        if abs_q > self.threshold:
            if state.contracts != 0:
                reward = state.exit_position()

            if q > 0:
                state.enter_long(amount)
                action = ACTION_LONG
            else:
                state.enter_short(amount)
                action = ACTION_SHORT

        elif abs_q < self.threshold and self.is_opposite_direction(q, state):
            reward = state.exit_position()
            action = ACTION_EXIT

        return (reward * self.intrinsic_fac, action)

    def is_opposite_direction(self, q: float, state: State) -> bool:
        return (q > 0 and state.contracts < 0) or (q < 0 and state.contracts > 0)
