import subprocess
import numpy as np
import pandas as pd


def get_git_root():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def check_all_float(df: pd.DataFrame, columns: list):
    for col in columns:
        if not df[col].dtype == np.float64:
            raise ValueError(f"{col} is not a float")
