
import contextlib
from enum import Enum

class ActionSpace(Enum):
    SIT = 0
    CLOSE = 1
    BUY = 2
    SELL = 3

class Action:
    __identifier: ActionSpace
    name: str
    value: int
    
    def __init__(self, type: ActionSpace | int | str):
        if isinstance(type, ActionSpace):
            self.__identifier = ActionSpace
        else:
            with contextlib.suppress(Exception):
                self.__identifier = ActionSpace[type]
            try:
                self.__identifier = ActionSpace(type)
            except Exception as e:
                raise ValueError(
                    f"Provided type ({type}) not part of the ActionSpace: {[(a.name, a.value) for a in ActionSpace]}"
                ) from e

        self.name = self.__identifier.name
        self.value = self.__identifier.value
            
