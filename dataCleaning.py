import pandas as pd


def csvCleaning(data):
    df = pd.read_csv(data)
    df.drop_duplicates(inplace=True) # Remove duplicates in the original DataFrame
    df.dropna(inplace=True)
    df = df[df['comment'].str.contains('.desktop') == False]
    df['comment'] = df['comment'].str.replace('\n', '')
    df['comment'] = df['comment'].str.replace('\t', '')
    df['comment'] = df['comment'].str.replace('[^\w\s]', '')
    df.to_csv('data/clean_data.csv')