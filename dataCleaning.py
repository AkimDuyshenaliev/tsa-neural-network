import pandas as pd


def csvCleaning(data):
    df = pd.read_csv(data)
    df.drop_duplicates(inplace=True) # Remove duplicates in the original DataFrame
    df.dropna(inplace=True)
    df = df[df['comment'].str.contains('.desktop') == False]
    df['comment'] = df['comment'].str.replace('[^\w\s]', ' ')
    df['comment'] = df['comment'].str.replace('''!"#$%&'()*+,-./:;<=>?@[]^_`{|}~''', ' ')
    df['comment'] = df['comment'].str.replace('\n\t', '')
    df['comment'] = df['comment'].str.lower()
    df['comment'] = df['comment'].str.strip()
    # df['comment'] = df['comment'].str.split(pat=' ')
    train_data = df.sample(frac = 0.7)
    test_data = df.drop(train_data.index)
    df.to_csv('data/clean_data.csv')
    train_data.to_csv('data/clean_train_data.csv')
    test_data.to_csv('data/clean_test_data.csv')


def freqVocab(data):
    df = pd.read_csv(data, sep='\t')
    df = df.drop(['PoS', 'R', 'D', 'Doc'], axis=1)
    df = df.sort_values(by='Freq(ipm)', ascending=False)
    df.insert(loc=0, column='index', value=range(1, len(df)+1))
    df.to_csv('data/freqVocabClean.csv', header=None, index=False)


def applyVocab(commentData, vocabData):
    c_df = pd.read_csv(commentData, usecols=['stars', 'comment'], index_col=False) # Comments DataFrame
    v_df = pd.read_csv(vocabData, index_col=False).set_index('index') # Vocabulary DatFrame

    c_df['comment'] = c_df['comment'].str.split(' ')
    print(c_df.head(15))
    print(v_df.head(15))