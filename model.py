import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.activations import softmax

from utils.utils import data_preprocessing, draw_plt


def rnn_model(data, tweetsData, ftModelData):
    (x_valid, y_valid), (x_test_valid, y_test_valid), ftEmbedding = data_preprocessing(
        data=data, 
        tweetsData=tweetsData,
        ftModelData=ftModelData)

    afunc = 'tanh'
    samples = 16
    model = keras.Sequential()

    if ftEmbedding == 0:
        model.add(layers.Embedding(10000, 64, input_length=10))
    else:
        model.add(layers.InputLayer(input_shape=x_valid.shape[1:]))

    model.add(layers.Bidirectional(layers.LSTM(samples*2, return_sequences=True, activation=afunc)))
    model.add(layers.Bidirectional(layers.LSTM(samples*2, activation=afunc)))
    model.add(layers.Dense(10))
    model.summary()

    model.compile(
        optimizer='Adam',
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
    (x_valid, y_valid), (x_test_valid, y_test_valid), ftEmbedding = data_preprocessing(
        data=data, 
        tweetsData=tweetsData,
        ftModelData=ftModelData) 

    ## CNN Model ###
    afunc = 'relu'
    samples = 16
    model = keras.models.Sequential()
    if ftEmbedding == 0:
        model.add(layers.Embedding(10000, 64, input_length=10))
    model.add(layers.Conv1D(samples, 3, activation=afunc, input_shape=(10, 100), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D())
    model.add(layers.Conv1D((samples*2), 3, activation=afunc, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D())
    model.add(layers.Conv1D((samples*4), 3, activation=afunc, padding='same'))

    model.add(layers.Flatten())
    model.add(layers.Dense((samples*8), activation=afunc))
    model.add(layers.Dense(10))

    model.summary()

    model.compile(
        optimizer='Adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    history = model.fit(
        x_valid, y_valid, 
        epochs=5,
        validation_data=(x_test_valid, y_test_valid))

    test_loss, test_acc = model.evaluate(x_test_valid,  y_test_valid, verbose=2)
    print(test_loss, test_acc)

    draw_plt(history=history)