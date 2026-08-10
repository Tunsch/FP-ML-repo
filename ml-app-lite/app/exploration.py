import pandas as pd

def run_exploration(train_df: pd.DataFrame):
    print(train_df.shape)
    print("Klassen", train_df["label"].unique)
    print("Sessions je Klasse:\n", train_df.groupby("label").nunique())
    train_df.head()
    train_df.describe()

    #Plots