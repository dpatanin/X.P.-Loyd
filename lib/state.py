import pandas as pd


class State:
    """
    This object represents one state, operations happen inplace.\n
    Ensures that data contains 'close' column.
    To feed it to a model use `data.to_numpy()`. (only data is used as model input)

    |`data`: Sequence of this state's price data as a pandas DataFrame.
    |`balance`: Current balance.
    |`entry_price`: Price with which position was entered.
    |`position`: Negative represents short position, positive long, 0 none.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        balance=0.00,
        entry_price=0.00,
        position=0,
    ):
        self.data = data
        self.assert_columns()

        self.balance = balance
        self.entry_price = entry_price
        self.position = position

    def enter_position(self, position: float):
        assert self.position == 0, "Exit current position first."
        self.entry_price = self.data["close"].iloc[-1]
        self.position = position

    def exit_position(self) -> float:
        profit = (self.data["close"].iloc[-1] - self.entry_price) * self.position
        self.balance += profit
        self.entry_price = 0
        self.position = 0

        return profit

    def assert_columns(self):
        req_columns = ["close"]
        if missing_columns := set(req_columns) - set(self.data.columns):
            raise ValueError(f"Sequence is missing required columns: {missing_columns}")

    def copy(self):
        return State(
            data=self.data,
            balance=self.balance,
            entry_price=self.entry_price,
            position=self.position,
        )

    def __str__(self):
        """
        Human readable DataFrame representation of the State object.
        """
        df = pd.DataFrame(self.data)
        df["position"] = [self.position] * len(df.index)
        df["entry price"] = [self.entry_price] * len(df.index)
        df["balance"] = [self.balance] * len(df.index)
        return df.__str__()
