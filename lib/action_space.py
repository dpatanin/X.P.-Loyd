from lib.constants import ACTION_EXIT, ACTION_LONG, ACTION_SHORT, ACTION_STAY, VOLUME
from lib.state import State


class ActionSpace:
    """
    Interprets the agent's prediction (q-values) and performs an action on a state (inplace).

    |`threshold`: The threshold of the prediction/value to act. (0 - 1); 0: Always act; 1: Never act.
    |`limit`: Absolute trading limit per single trade.

    Strategy:\n
    If under threshold, do nothing unless prediction opposes current position, in that case be careful and exit position.
    If above threshold, enter the predicted position. If already in that position, keep your contracts and reenter.
    """

    def __init__(self, threshold: float, limit: int):
        self.threshold = threshold
        self.limit = limit

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
        Returns tuple with profit and taken action.
        """
        action = ACTION_STAY
        profit = 0.00
        abs_q = abs(q)
        amount = self.calc_trade_amount(q, state)

        if abs_q > self.threshold:
            if state.contracts != 0:
                profit = state.exit_position()

            if q > 0:
                state.enter_long(amount)
                action = ACTION_LONG
            else:
                state.enter_short(amount)
                action = ACTION_SHORT

        elif abs_q < self.threshold and self.is_opposite_direction(q, state):
            profit = state.exit_position()
            action = ACTION_EXIT

        return (profit, action)

    def is_opposite_direction(self, q: float, state: State) -> bool:
        return (q > 0 and state.contracts < 0) or (q < 0 and state.contracts > 0)
