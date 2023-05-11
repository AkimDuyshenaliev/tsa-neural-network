import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import CSVLogger

from utils.utils import data_preprocessing, draw_plt, coloring


def rnn_model(data, tweetsData, ftModelData, logsPath, chosenAction=False):
    (x_valid, y_valid), (x_test_valid, y_test_valid), forFT, vocab_size = data_preprocessing(
        data=data, 
        tweetsData=tweetsData,
        ftModelData=ftModelData,
        chosenAction=chosenAction)

    units = 16
    model = keras.Sequential()

    if forFT is False:
        model.add(layers.Embedding(vocab_size, units*2, input_length=x_valid.shape[1]))
    else:
        model.add(layers.InputLayer(input_shape=x_valid.shape[1:]))

    model.add(layers.Bidirectional(layers.LSTM(
        units=units*2, 
        return_sequences=True, 
        dropout=0.2, 
        recurrent_dropout=0.2)))
    model.add(layers.Bidirectional(layers.LSTM(
        units=units, 
        dropout=0.2, 
        recurrent_dropout=0.2)))
    model.add(layers.Dense(units*2, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(
        optimizer='rmsprop',
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy'])

    csv_logger = CSVLogger(f'{logsPath}rnn_embedding-option-{chosenAction}.log')

    history = model.fit(
        x_valid, y_valid, 
        epochs=5,
        batch_size=32,
        validation_data=(x_test_valid, y_test_valid),
        callbacks=[csv_logger])

    test_loss, test_acc = model.evaluate(x_test_valid,  y_test_valid, verbose=2)
    print(test_loss, test_acc)

    draw_plt(history=history, name=f'rnn-embedding-option-{chosenAction}')


def cnn_model(data, tweetsData, ftModelData, logsPath, chosenAction=False):
    (x_valid, y_valid), (x_test_valid, y_test_valid), forFT, vocab_size = data_preprocessing(
        data=data, 
        tweetsData=tweetsData,
        ftModelData=ftModelData, 
        chosenAction=chosenAction)

    ## CNN Model ###
    afunc = 'relu'
    filters = 16
    model = keras.models.Sequential()

    if forFT is False: model.add(layers.Embedding(vocab_size, filters, input_length=x_valid.shape[1]))
    else:
        model.add(layers.InputLayer(input_shape=x_valid.shape[1:]))

    model.add(layers.Conv1D(filters=filters*4, kernel_size=3, activation=afunc, padding='same'))
    model.add(layers.Dropout(0.3))
    model.add(layers.MaxPooling1D())
    model.add(layers.Conv1D(filters=(filters*2), kernel_size=4, activation=afunc, padding='same'))
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling1D())
    model.add(layers.Conv1D(filters=(filters), kernel_size=5, activation=afunc, padding='same'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Flatten())
    model.add(layers.Dense((filters*8), activation=afunc))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(
        optimizer='rmsprop',
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy'])

    csv_logger = CSVLogger(f'{logsPath}cnn_embedding-option-{chosenAction}.log')

    history = model.fit(
        x_valid, y_valid, 
        epochs=5,
        batch_size=32,
        validation_data=(x_test_valid, y_test_valid),
        callbacks=[csv_logger])

    test_loss, test_acc = model.evaluate(x_test_valid,  y_test_valid, verbose=2)
    print(test_loss, test_acc)

    draw_plt(history=history, name=f'cnn-embedding-option-{chosenAction}')


def runTest(comments, tweets, ftModels, logs):
    print('Run test function')
    possibleActions = [1, 2, 3, 4]
    for chosenAction in possibleActions:
        models = [ 
            {'name': 'Run CNN model',
            'func': lambda: cnn_model(
                    data=comments,
                    tweetsData=tweets,
                    ftModelData=ftModels,
                    logsPath=logs,
                    chosenAction=chosenAction)},
            {'name': 'Run RNN model',
            'func': lambda: rnn_model(
                    data=comments,
                    tweetsData=tweets,
                    ftModelData=ftModels,
                    logsPath=logs,
                    chosenAction=chosenAction)},
        ]
        for model in models:
            temp = '\nRunning %s' % model['name'][4:]
            coloring(data=temp, r='38', g='05', b='46')
            model['func']()

