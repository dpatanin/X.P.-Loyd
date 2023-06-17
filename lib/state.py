import pandas as pd


class State:
    """
    This object represents one state.\n
    Ensures that data contains 'Close' & 'Volume' columns.
    Entering positions modifies this state. To feed it to a model use the `to_numpy()` method.

    |`data`: Sequence of this state's price data as a pandas DataFrame.
    |`balance`: Current balance.
    |`entry_price`: Price with which position was entered.
    |`contracts`: Amount of contracts if in position. Negative represents short position, positive long.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        balance=0.00,
        entry_price=0.00,
        contracts=0,
    ):
        self.data = data
        self.assert_columns()

        self.balance = balance
        self.entry_price = entry_price
        self.contracts = contracts

    def enter_long(self, contracts: int, price_per_contract: float):
        self.assert_valid_operation(contracts, price_per_contract)
        self.entry_price = self.data["close"].iloc[-1] if contracts > 0 else 0.00
        self.balance -= contracts * price_per_contract
        self.contracts = contracts

    def enter_short(self, contracts: int, price_per_contract: float):
        self.assert_valid_operation(contracts, price_per_contract)
        self.entry_price = self.data["close"].iloc[-1] if contracts > 0 else 0.00
        self.balance += contracts * price_per_contract
        self.contracts = -contracts

    def exit_position(self, price_per_contract: float) -> float:
        assert self.has_position(), "No position to exit."
        profit = (
            (self.data["close"].iloc[-1] - self.entry_price)
            * self.contracts
            * price_per_contract
        )
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

    def to_df(self) -> pd.DataFrame:
        """
        Human readable Dataframe representation of the State object.
        """
        df = pd.DataFrame(self.data)
        df["contracts"] = self.contracts
        df["entryPrice"] = self.entry_price
        df["balance"] = self.balance
        return df

    def to_numpy(self):
        """
        Machine readable Numpy representation of the State object.
        """
        return self.to_df().to_numpy()

    def assert_columns(self):
        req_columns = ["close", "volume"]
        if missing_columns := set(req_columns) - set(self.data.columns):
            raise ValueError(f"Sequence is missing required columns: {missing_columns}")

    def assert_valid_operation(self, contracts: int, price_per_contract: float):
        assert (
            not self.has_position()
        ), f"Exit current position first. Current position: {self.has_position()}."
        assert (
            contracts >= 0
        ), f"Invalid amount of contracts provided. Received: {contracts}."
        assert (
            price_per_contract > 0
        ), f"Invalid price per contract provided. Received: {price_per_contract}."

    def __str__(self):
        return self.to_df().__str__()
