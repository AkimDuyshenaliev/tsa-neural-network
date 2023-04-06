import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, GRU, Bidirectional, Dense, Embedding, Dropout
from keras.datasets import imdb
from utils.utils import coloring


def text_sentiment_neural_network(testData, trainData):
    testData = pd.read_csv(testData)
    trainData = pd.read_csv(trainData)
    vocab_size = 5000
    max_words = 400

    word_ind = imdb.get_word_index()
    word_ind = {i: word for word, i in word_ind.items()}

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

    print([word_ind[i] for i in x_train[0]])
    print("Max length of a review:: ", len(max((x_train + x_test), key=len)))
    print("Min length of a review:: ", len(min((x_train + x_test), key=len)))

    x_train = keras.utils.pad_sequences(x_train, maxlen=max_words)
    x_test = keras.utils.pad_sequences(x_test, maxlen=max_words)

    x_valid, y_valid = x_train[:64], y_train[:64]
    x_train_, y_train_ = x_train[64:], y_train[64:]

    print(f'Valid x - {x_valid}')

    # For personaly collected data
    # measure = np.vectorize(len)
    # maxlen = 'max len of comments %s' % measure(testdata['comment'].astype(str)).max(axis=0)
    # minlen = 'min len of comments %s' % measure(testdata['comment'].astype(str)).min(axis=0)
    # coloring(data=maxlen)
    # coloring(data=minlen)

    # model = sequential(name='bidirectional_lstm')
    # model.add(Embedding(vocab_size, ))
    # End
