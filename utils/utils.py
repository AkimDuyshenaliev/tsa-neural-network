import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import fasttext

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.datasets import imdb


def color(r="38", g="05", b="222"):
    def my_docorator(func):
        def wrapper(self=None):
            print(f"\033[01;{r};{g};{b}m", end="")
            func() if not self else func(self)
            print("\033[0m")
        return wrapper
    return my_docorator


def coloring(data, r="38", g="05", b="222"):
    color(r, g, b)(lambda: print(data, end=""))()


def draw_plt(history, name):
    def plot_graphs(history, metric):
        plt.plot(history.history[metric])
        plt.plot(history.history['val_'+metric], '')
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend([metric, 'val_'+metric])

    plt.figure(figsize=(16, 6))
    plt.subplot(2, 2, 2)
    plot_graphs(history, 'accuracy')
    plt.subplot(2, 2, 1)
    plot_graphs(history, 'loss')

    plt.subplot(2, 1, 2)
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Validated accuracy')
    plt.legend(loc='lower right')

    format = 'png'
    # plt.show()
    plt.savefig(f'data/logs/matplot-output/{name}.{format}', format=format)


def export_for_fasttext(df):
    ### Prepare data for fasttext supervised training
    df_ft = df.set_axis(['value', 'data'], axis=1).reset_index(drop=True)
    df_ft['value'] = df_ft['value'].apply(lambda string: 'negative' if string == 0 else 'positive')
    df_ft['value'] = '__label__' + df_ft['value'].astype(str)
    df_ft['text'] = df_ft['value'] + ' ' + df_ft['data']
    df_ft.drop(['value', 'data'], axis=1,inplace=True)
    df_ft.to_csv('data/ft_training_data.txt', header=None, index=False)
    df_ft.to_string('data/test.txt')

    coloring(data='Successfully exported', r='38', g='05', b='46')
    ### End of preparing data for fasttext supervised training


def match_words_with_numbers(df, max_len, vocab_size, forFT):
    num_words = vocab_size
    oov_token = '<UNK>'
    pad_type = 'post'
    trunc_type = 'post'
    maxlen = max_len

    dfTrain = df.sample(frac=0.7, random_state=123)
    train_data = dfTrain[1].tolist()
    y_valid = dfTrain[0].to_numpy().astype('int8').flatten()

    dfTest = df.drop(dfTrain.index)
    test_data = dfTest[1].tolist()
    y_test_valid = dfTest[0].to_numpy().astype('int8').flatten()

    # Tokenize our training data
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(train_data)

    # Encode training data sentences into sequences
    train_sequences = tokenizer.texts_to_sequences(train_data)

    # Pad the training sequences
    x_valid = pad_sequences(train_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

    test_sequences = tokenizer.texts_to_sequences(test_data)
    x_test_valid = pad_sequences(test_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

    return (x_valid, y_valid), (x_test_valid, y_test_valid), forFT, vocab_size


def imdb_data(max_len, vocab_size, forFT=False):
    (x_train, y_valid), (x_test, y_test_valid) = imdb.load_data(num_words=vocab_size)

    if forFT is True:
        # Use the default parameters to keras.datasets.imdb.load_data
        start_char = 1
        oov_char = 2
        index_from = 3

        # Retrieve the word index file mapping words to indices
        word_index = imdb.get_word_index()

        # Reverse the word index to obtain a dict mapping indices to words
        # And add `index_from` to indices to sync with `x_train`
        inverted_word_index = dict(
            (i + index_from, word) for (word, i) in word_index.items()
        )
        # Update `inverted_word_index` to include `start_char` and `oov_char`
        inverted_word_index[start_char] = ""
        inverted_word_index[oov_char] = ""

        x_train = pd.Series(x_train)
        x_test = pd.Series(x_test)

        # Decode the first sequence in the dataset
        x_valid = x_train.apply(lambda sentence: " ".join(inverted_word_index[i] for i in sentence))
        x_test_valid = x_test.apply(lambda sentence: " ".join(inverted_word_index[i] for i in sentence))
    else:
        x_valid = pad_sequences(x_train, maxlen=max_len)
        x_test_valid = pad_sequences(x_test, maxlen=max_len)

    return (x_valid, y_valid), (x_test_valid, y_test_valid), forFT, vocab_size


def data_preprocessing(data, tweetsData, ftModelData, chosenAction=False):
    '''
    0 - negative
    1 - positive
    '''

    max_len = 10
    min_len = 4
    vocab_size = 10000

    tweetsPos = pd.read_csv(tweetsData[1], skiprows=1, header=None, usecols=[2])
    tweetsNeg = pd.read_csv(tweetsData[0], skiprows=1, header=None, usecols=[2])
    tweetsPos.insert(loc=0, column=0, value=1) # Add value of "1" for every row
    tweetsNeg.insert(loc=0, column=0, value=0) # Add value of "0" for every row

    train_comments = pd.read_csv(data[1], skiprows=1, header=None, usecols=[2, 3])
    train_comments[2] = train_comments[2].apply(lambda num: 1 if num > 3 else 0)

    test_comments = pd.read_csv(data[2], skiprows=1, header=None, usecols=[2, 3])
    test_comments[2] = test_comments[2].apply(lambda num: 1 if num > 3 else 0)

    sentences = pd.concat([train_comments[3], test_comments[3], tweetsNeg[2], tweetsPos[2]], axis=0)
    value = pd.concat([train_comments[2], test_comments[2], tweetsNeg[0], tweetsPos[0]], axis=0)

    df = pd.concat([value, sentences], axis=1).reset_index(drop=True)

    # Prepare data for FastText embedding
    def ft_prepare_data(df, forFT, vocab_size):
        dfTrain = df.sample(frac=0.7, random_state=123)
        dfTest = df.drop(dfTrain.index)

        (x_train, y_train) = dfTrain.iloc[:, 1], dfTrain.iloc[:, 0]
        (x_test, y_test) = dfTest.iloc[:, 1], dfTest.iloc[:, 0]

        y_train = y_train.to_numpy()
        y_valid = y_train.astype('int8').flatten()

        y_test = y_test.to_numpy()
        y_test_valid = y_test.astype('int8').flatten()
        return (x_train, y_valid), (x_test, y_test_valid), forFT, vocab_size 

    options = [
        {'name': 'Exit'},
        {'name': 'Custom Dataset, Tensorflow embedding',
         'func': lambda: match_words_with_numbers(
                    df=df, 
                    max_len=max_len, 
                    vocab_size=vocab_size,
                    forFT=False)},
        {'name': 'Custom Dataset, FastText embedding',
         'func': lambda: ft_prepare_data(
                    df=df,
                    vocab_size=vocab_size,
                    forFT=True)},
        {'name': 'IMDB Dataset, Tensorflow embedding',
         'func': lambda: imdb_data(
                    max_len=max_len, 
                    vocab_size=vocab_size,
                    forFT=False)},
        {'name': 'IMDB Dataset, FastText embedding',
         'func': lambda: imdb_data(
                    max_len=max_len, 
                    vocab_size=vocab_size,
                    forFT=True)},
        {'name': 'Export for FastText training',
         'func': lambda: export_for_fasttext(
                    df=df)},
    ]
    try:
        if chosenAction is False:
            print('\n')
            {print(f"{key}: {option['name']}") for key, option in enumerate(options)}
            if (choice := int(input('\nChoose what to do: '))) == 0:
                print('Exiting\n')
                return
        else:
            choice = chosenAction
    except ValueError:
        print('Exiting\n')
        return
    except IndexError:
        print('No such option\n')
        return
    
    print('Chosen - %s' % options[choice]['name'])
    (x_train, y_valid), (x_test, y_test_valid), forFT, vocab_size = options[choice]['func']()

    # If FastText embedding was not chosen then return data
    if forFT is False:
        print(f'Train data: {x_train.shape}, {y_valid.shape}')
        print(f'Test data:  {x_test.shape}, {y_test_valid.shape}')
        print(f'FastTest embedding: {forFT}')
        return (x_train, y_valid), (x_test, y_test_valid), forFT, vocab_size

    x_train = x_train.str.split(pat=' ')
    x_test = x_test.str.split(pat=' ')

    ### Normalizing data and embedding using fasttext
    def dataNormalizaition(data, max_len):
        data = data[:max_len]
        for i in range(max_len):
            if i < len(data):
                data[i] = ftModel.get_word_vector(data[i]).astype('float32')
            else:
                data.append(np.zeros(shape=100, dtype=np.float32))
        return np.stack(data, axis=0)

    ftModel = fasttext.load_model(ftModelData[0] if choice != 4 else ftModelData[1])
    x_train = x_train.apply(lambda string: dataNormalizaition(data=string, max_len=max_len))
    x_test = x_test.apply(lambda string: dataNormalizaition(data=string, max_len=max_len))

    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()
    
    x_valid = np.empty((x_train.shape[0], x_train[0].shape[0], x_train[0].shape[1])).astype('float32')
    for i, sentence in enumerate(x_train):
        x_valid[i] = sentence

    x_test_valid = np.empty((x_test.shape[0], x_test[0].shape[0], x_test[0].shape[1])).astype('float32')
    for i, sentence in enumerate(x_test):
        x_test_valid[i] = sentence

    print(f'Train data: {x_valid.shape}, {x_test_valid.shape}')
    print(f'Test data:  {y_valid.shape}, {y_test_valid.shape}')
    print(f'FastTest embedding: {forFT}')
    # ### End of embedding using fasttext

    return (x_valid, y_valid), (x_test_valid, y_test_valid), forFT, vocab_size