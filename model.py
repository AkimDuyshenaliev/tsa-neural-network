import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import fasttext
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils.utils import coloring


def textSentiment_RNN_neuralNetwork(data, tweetsData, ftModelData):
    '''
    0 - negative
    1 - positive
    '''
    tweetsPos = pd.read_csv(tweetsData[1], skiprows=1, header=None, usecols=[2])
    tweetsNeg = pd.read_csv(tweetsData[0], skiprows=1, header=None, usecols=[2])
    tweetsPos.insert(loc=0, column=0, value=1)
    tweetsNeg.insert(loc=0, column=0, value=0)

    train_comments = pd.read_csv(data[1], skiprows=1, header=None, usecols=[2, 3])
    test_comments = pd.read_csv(data[2], skiprows=1, header=None, usecols=[2, 3])

    sentences = pd.concat([train_comments[3], test_comments[3], tweetsNeg[2], tweetsPos[2]], axis=0)
    value = pd.concat([train_comments[2], test_comments[2], tweetsNeg[0], tweetsPos[0]], axis=0)

    df = pd.concat([value, sentences], axis=1)
    dfTrain = df.sample(frac=0.7, random_state=123)
    dfTest = df.drop(dfTrain.index)

    (x_train, y_train) = dfTrain.iloc[:, 1], dfTrain.iloc[:, 0]
    (x_test, y_test) = dfTest.iloc[:, 1], dfTest.iloc[:, 0]

    x_train = x_train.str.split(pat=' ')
    x_test = x_test.str.split(pat=' ')

    y_train = y_train.apply(lambda num: 1 if num > 3 else 0)
    y_test = y_test.apply(lambda num: 1 if num > 3 else 0)

    max_len = 10
    min_len = 4

    def dataNormalizaition(data, max_len):
        data = data[:10]
        for i in range(max_len):
            if i < len(data):
                data[i] = ftModel.get_word_vector(data[i])
            else:
                data.append(ftModel.get_word_vector('_'))
        return np.stack(data, axis=0)

    ftModel = fasttext.load_model(ftModelData)
    x_train = x_train.apply(lambda string: dataNormalizaition(data=string, max_len=max_len))

    x_train = x_train.to_numpy()
    x_valid = np.empty((x_train.shape[0], x_train[0].shape[0], x_train[0].shape[1]))
    for i, sentence in enumerate(x_train):
        x_valid[i] = sentence
