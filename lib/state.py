import pandas as pd

from lib.constants import BALANCE, CLOSE, CONTRACTS, ENTRY_PRICE


class State:
    """
    This object represents one state.\n
    Ensures that data contains 'close' column.
    Entering positions modifies this state. To feed it to a model use the `to_numpy()` method.

    |`data`: Sequence of this state's price data as a pandas DataFrame.
    |`balance`: Current balance.
    |`entry_price`: Price with which position was entered.
    |`contracts`: Amount of contracts if in position. Negative represents short position, positive long.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tick_size: float,
        tick_value: float,
        balance=0.00,
        entry_price=0.00,
        contracts=0,
    ):
        self.data = data
        self.assert_columns()

        self.tick_size = tick_size
        self.tick_value = tick_value
        self.balance = balance
        self.entry_price = entry_price
        self.contracts = contracts

    def enter_long(self, contracts: int):
        self.assert_valid_operation(contracts)
        self.entry_price = self.data[CLOSE].iloc[-1]
        self.contracts = contracts

    def enter_short(self, contracts: int):
        self.assert_valid_operation(contracts)
        self.entry_price = self.data[CLOSE].iloc[-1]
        self.contracts = -contracts

    def exit_position(self) -> float:
        assert self.contracts != 0, "No position to exit."
        profit = (
            ((self.data[CLOSE].iloc[-1] - self.entry_price) / self.tick_size)
            * self.tick_value
            * self.contracts
        )
        self.balance += profit
        self.entry_price = 0
        self.contracts = 0

        return profit

    def to_df(self) -> pd.DataFrame:
        """
        Human readable DataFrame representation of the State object.
        """
        df = pd.DataFrame(self.data)
        df[CONTRACTS] = [self.contracts] * len(df.index)
        df[ENTRY_PRICE] = [self.entry_price] * len(df.index)
        df[BALANCE] = [self.balance] * len(df.index)
        return df

    def to_numpy(self):
        """
        Machine readable Numpy representation of the State object.
        """
        return self.to_df().to_numpy()

    def assert_columns(self):
        req_columns = [CLOSE]
        if missing_columns := set(req_columns) - set(self.data.columns):
            raise ValueError(f"Sequence is missing required columns: {missing_columns}")

    def assert_valid_operation(self, contracts: int):
        assert self.contracts == 0, "Exit current position first."
        assert (
            contracts > 0
        ), f"Invalid amount of contracts provided. Received: {contracts}."

    def copy(self):
        return State(
            data=self.data,
            tick_size=self.tick_size,
            tick_value=self.tick_value,
            balance=self.balance,
            entry_price=self.entry_price,
            contracts=self.contracts,
        )

    def __str__(self):
        return self.to_df().__str__()
