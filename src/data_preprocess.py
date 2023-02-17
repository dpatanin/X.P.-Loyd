import pandas as pd
import os
from helper import assert_columns


def load_data(file_path: str) -> pd.DataFrame:
    assert os.path.exists(file_path), f"{file_path} does not exist."
    data = pd.read_csv(file_path)
    assert_columns(data)
    
    return data
