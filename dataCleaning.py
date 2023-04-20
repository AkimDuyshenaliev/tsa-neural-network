from utils.utils import coloring
import pandas as pd


def columnCleaning(df, col):
    try:
        min_len = int(input('Select minimum length of each comment: '))
    except:
        min_len = 4

    col = df.columns.get_loc(col) if type(col) is str else col

    df.iloc[:, col] = df.iloc[:, col].str.replace('[\n\t]', ' ').str.replace('[^\w\s]', ' ')
    df.iloc[:, col] = df.iloc[:, col].str.lower().str.strip().str.split(pat=' ')
    df.iloc[:, col] = df.iloc[:, col] \
        .apply(lambda string : ' '.join(s for s in string if s.isalnum())).str.split(pat=' ')
    df.drop(df[df.iloc[:, col].str.len() < min_len].index, inplace=True) # Drop comments shorter than 4 words
    df.iloc[:, col] = df.iloc[:, col].apply(lambda data : ' '.join(s for s in data))

    return df


def tweetsCleaning(data, clean_data_path):
    neg_df = pd.read_csv(data[0], sep=';', usecols=[2, 3])
    pos_df = pd.read_csv(data[1], sep=';', usecols=[2, 3])
    
    neg_df = columnCleaning(df=neg_df, col=1)
    pos_df = columnCleaning(df=pos_df, col=1)

    neg_df.to_csv(f'{clean_data_path}/tweets-clean/clean_tweets_negative.csv')
    pos_df.to_csv(f'{clean_data_path}/tweets-clean/clean_tweets_positive.csv')
    print('Done and written into CSV')


def csvCleaning(data, clean_data_path):
    try:
        min_len = int(input('Select minimum length of each comment: '))
    except:
        min_len = 4

    df = pd.read_csv(data)
    # Remove duplicates in the original DataFrame
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df = df[df['comment'].str.contains('.desktop') == False]
    
    df = columnCleaning(df=df, col='comment')

    train_data = df.sample(frac=0.7)
    test_data = df.drop(train_data.index)

    choice = int(input('1 - Print\n2 - Write to csv\n'))
    if choice == 1:
        print(df[75:100])
    elif choice == 2:
        ### Write to csv
        df.to_csv(f'{clean_data_path}/clean_comments.csv')
        train_data.to_csv(f'{clean_data_path}/clean_train_comments.csv')
        test_data.to_csv(f'{clean_data_path}/clean_test_comments.csv')
        coloring(data='Cleaning finished, files created', r='38', g='05', b='46')
    else:
        print('No such thing, printing out')
        print(df[75:100])


def makeTrainSeries(commentData, tweetsData):
    full_comments_df = pd.read_csv(commentData[0], skiprows=1, header=None, usecols=[3])
    # train_comments_df = pd.read_csv(commentData[1], skiprows=1, header=None, usecols=[3])
    # test_comments_df = pd.read_csv(commentData[2], skiprows=1, header=None, usecols=[3])

    neg_tweets_df = pd.read_csv(tweetsData[0], header=None, usecols=[2])
    pos_tweets_df = pd.read_csv(tweetsData[1], header=None, usecols=[2])

    main_df = pd.concat\
        ([full_comments_df.iloc[:, 0], neg_tweets_df.iloc[:, 0], pos_tweets_df.iloc[:, 0]])\
            .to_csv('data/train-data/unsupervised_train_data.csv', index=False, header=None)
    print('Done\n')