import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np

import fasttext
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

from utils.utils import coloring


def draw_plt(history):
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    plt.show()


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


def data_preprocessing(data, tweetsData, ftModelData, functionCall=False):
    '''
    0 - negative
    1 - positive
    '''
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
    dfTrain = df.sample(frac=0.7, random_state=123)
    dfTest = df.drop(dfTrain.index)

    if functionCall is True:
        pass
    else:
        if int(input('Export to FastText?\n[1=yes, 0=no] -> ')) == 1:
            export_for_fasttext(df)

    (x_train, y_train) = dfTrain.iloc[:, 1], dfTrain.iloc[:, 0]
    (x_test, y_test) = dfTest.iloc[:, 1], dfTest.iloc[:, 0]

    x_train = x_train.str.split(pat=' ')
    x_test = x_test.str.split(pat=' ')

    y_train = y_train.to_numpy()
    y_valid = y_train.astype('int8').flatten()

    y_test = y_test.to_numpy()
    y_test_valid = y_test.astype('int8').flatten()

    max_len = 10
    min_len = 4

    ### Normalizing data and embedding using fasttext
    def dataNormalizaition(data, max_len):
        data = data[:10]
        for i in range(max_len):
            if i < len(data):
                data[i] = ftModel.get_word_vector(data[i]).astype('float32')
            else:
                data.append(np.zeros(shape=100, dtype=np.float32))
        return np.stack(data, axis=0)

    ftModel = fasttext.load_model(ftModelData)
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

    unique, counts = np.unique(y_valid, return_counts=True)
    print(dict(zip(unique, counts)))
    # ### End of embedding using fasttext

    return (x_valid, y_valid), (x_test_valid, y_test_valid)


def rnn_model(data, tweetsData, ftModelData):
    (x_valid, y_valid), (x_test_valid, y_test_valid) = data_preprocessing(
        data=data, 
        tweetsData=tweetsData,
        ftModelData=ftModelData, 
        functionCall=True)

    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=x_valid.shape[1:]))
    model.add(layers.LSTM(128))
    model.add(layers.Dense(10))
    model.summary()

    model.compile(
        optimizer='sgd',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    history = model.fit(
        x_valid, y_valid, 
        epochs=5,
        validation_data=(x_test_valid, y_test_valid))
    test_loss, test_acc = model.evaluate(x_test_valid,  y_test_valid, verbose=2)
    print(test_loss, test_acc)

    draw_plt(history=history)


def ft_cnn_model(data, tweetsData, ftModelData):
    (x_valid, y_valid), (x_test_valid, y_test_valid) = data_preprocessing(
        data=data, 
        tweetsData=tweetsData,
        ftModelData=ftModelData, 
        functionCall=True)

    ## CNN Model ###
    samples = 16
    model = keras.models.Sequential()
    model.add(layers.Conv1D(samples, 3, activation='relu', input_shape=(10, 100), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D())
    model.add(layers.Conv1D((samples*2), 3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D())
    model.add(layers.Conv1D((samples*4), 3, activation='relu', padding='same'))

    model.add(layers.Flatten())
    model.add(layers.Dense((samples*8), activation='relu'))
    model.add(layers.Dense(10))

    model.summary()

    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    history = model.fit(
        x_valid, y_valid, 
        epochs=5,
        validation_data=(x_test_valid, y_test_valid))

    test_loss, test_acc = model.evaluate(x_test_valid,  y_test_valid, verbose=2)
    print(test_loss, test_acc)

    draw_plt(history=history)