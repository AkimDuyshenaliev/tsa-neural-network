import pandas as pd

def csvCleaning(data):
    df = pd.read_csv(data)
    print(df)