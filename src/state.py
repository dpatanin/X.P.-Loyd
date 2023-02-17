import pandas as pd
from src.helper import assert_columns, default_header


class State:
    def __init__(
        self,
        data: pd.DataFrame,
        balance=10000.00,
        entry_price=0.00,
        contracts=0,
    ):
        assert_columns(data, default_header)

        self.data = data
        self.balance = balance
        self.entry_price = entry_price
        self.contracts = contracts

    def enter_long(self, entry_price: float, contracts: int, price_per_contract: float):
        assert not self.has_position(), "Exit current position first."
        self.entry_price = entry_price
        self.balance -= contracts * price_per_contract
        self.contracts = contracts

    def enter_short(
        self, entry_price: float, contracts: int, price_per_contract: float
    ):
        assert not self.has_position(), "Exit current position first."
        self.entry_price = entry_price
        self.balance += contracts * price_per_contract
        self.contracts = contracts

    def exit_position(self, exit_price: float, price_per_contract: float) -> float:
        assert self.has_position(), "No position to exit."
        profit = (exit_price - self.entry_price) * self.contracts * price_per_contract
        self.balance += profit
        self.entry_price = 0
        self.contracts = 0

        return profit

    def has_position(self):
        """
        return: 1 for long position, -1 for short position, 0 for no position
        """
        if self.contracts > 0:
            return 1
        elif self.contracts < 0:
            return -1
        else:
            return 0

    def rep_position(self):
        if not self.has_position():
            return "No position."

        position = self.has_position()
        return f"{'Long' if position == 1 else 'Short'} position: {self.contracts} contracts entered at {self.entry_price}."

    def to_df(self) -> pd.DataFrame:
        """
        Human readable representation.
        """
        return pd.DataFrame(
            {
                "data": [self.data.to_numpy()],
                "balance": [self.balance],
                "entry_price": [self.entry_price],
                "contracts": [self.contracts],
            }
        )

    def to_numpy(self):
        """
        Machine readable representation.
        """
        return self.to_df().to_numpy().ravel()

    def __str__(self):
        return self.to_df().__str__
