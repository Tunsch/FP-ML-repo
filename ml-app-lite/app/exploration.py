import pandas as pd

def run_exploration(train_df: pd.DataFrame):
    train_df.describe()
    train_df.head()