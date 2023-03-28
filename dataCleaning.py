import pandas as pd


def csvCleaning(data):
    df = pd.read_csv(data)
    df.drop_duplicates(inplace=True) # Remove duplicates in the original DataFrame
    df.dropna(inplace=True)
    df = df[df['comment'].str.contains('.desktop') == False]
    df['comment'] = df['comment'].str.replace('\n', '')
    df['comment'] = df['comment'].str.replace('\t', '')
    df['comment'] = df['comment'].str.replace('[^\w\s]', '')
    train_data = df.sample(frac = 0.7)
    test_data = df.drop(train_data.index)
    df.to_csv('data/clean_data.csv')
    train_data.to_csv('data/clean_train_data.csv')
    test_data.to_csv('data/clean_test_data.csv')