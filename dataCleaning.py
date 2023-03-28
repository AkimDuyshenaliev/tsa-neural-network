import pandas as pd


def csvCleaning(data):
    df = pd.read_csv(data)
    df.drop_duplicates(inplace=True) # Remove duplicates in the original DataFrame
    df.dropna(inplace=True)
    df = df[df['comment'].str.contains('.desktop') == False]
    df = df[df['comment'].str.encode('UTF-8', errors='ignore').str.decode('UTF-8')]
    df.to_csv('data/clean_data.csv')
    print(df)