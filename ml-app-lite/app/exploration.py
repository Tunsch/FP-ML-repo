import pandas as pd

def run_exploration(train_df: pd.DataFrame, feature_cols: list[str]):

    print("\nStatistik:")
    print(train_df[feature_cols].describe().T[["mean", "std", "min", "50%", "max"]])

    #Plots