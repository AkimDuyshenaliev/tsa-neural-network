from utils.utils import coloring
import pandas as pd


def csvCleaning(data, min_len=4):
    df = pd.read_csv(data)
    # Remove duplicates in the original DataFrame
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df = df[df['comment'].str.contains('.desktop') == False]
    
    ### Cleaning
    df['comment'] = df['comment'].str.replace('[\n\t]', ' ').str.replace('[^\w\s]', ' ')
    df['comment'] = df['comment'].str.lower().str.strip().str.split(pat=' ')
    df['comment'] = df['comment'] \
        .apply(lambda string : ' '.join(s for s in string if s.isalnum())).str.split(pat=' ')
    df.drop(df[df['comment'].str.len() < min_len].index, inplace=True) # Drop comments shorter than 4 words
    df['comment'] = df['comment'].apply(lambda data : ' '.join(s for s in data))
    ### End of cleaning

    train_data = df.sample(frac=0.7)
    test_data = df.drop(train_data.index)

    ### Write to csv
    df.to_csv('data/clean_data.csv')
    train_data.to_csv('data/clean_train_data.csv')
    test_data.to_csv('data/clean_test_data.csv')

    coloring(data='Cleaning finished, files created', r='38', g='05', b='46')


def freqVocab(data):
    df = pd.read_csv(data, sep='\t')
    df = df.drop(['PoS', 'R', 'D', 'Doc'], axis=1)
    df = df.sort_values(by='Freq(ipm)', ascending=False)
    df.insert(loc=0, column='index', value=range(1, len(df) + 1))
    df.to_csv('data/freqVocabClean.csv', header=None, index=False)


def applyVocab(commentData, vocabData):
    c_df = pd.read_csv(commentData, usecols=[
                       'stars', 'comment'], index_col=False)  # Comments DataFrame
    v_df = pd.read_csv(vocabData, index_col=False).set_index(
        'index')  # Vocabulary DatFrame

    c_df['comment'] = c_df['comment'].str.split(' ')
    print(c_df.head(15))
    print(v_df.head(15))
