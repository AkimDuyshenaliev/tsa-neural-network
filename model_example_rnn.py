from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Bidirectional, Dense, Embedding
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
import numpy as np


def example_RNN():
    vocab_size = 5000
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    
    print(x_train[0])

    word_idx = imdb.get_word_index()
    word_idx = {i: word for word, i in word_idx.items()}
    
    # print([word_idx[i] for i in x_train[0]])

    # print("Max length of a review:: ", len(max((x_train+x_test), key=len)))
    # print("Min length of a review:: ", len(min((x_train+x_test), key=len)))

    max_words = 400
    
    x_train = sequence.pad_sequences(x_train, maxlen=max_words)
    x_test = sequence.pad_sequences(x_test, maxlen=max_words)
    
    x_valid, y_valid = x_train[:64], y_train[:64]
    x_train_, y_train_ = x_train[64:], y_train[64:]

    print(x_valid)
    print(type(x_valid))
    print(x_valid.shape)
    print(y_valid)