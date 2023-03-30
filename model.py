import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from utils.utils import coloring


def text_sentiment_neural_network(testData, trainData):
    testData = pd.read_csv(testData)
    trainData = pd.read_csv(trainData)

    data = pd.concat([trainData['stars'], trainData['comment']], axis=1)
    coloring(data=data.shape)
    
    model = Sequential()

    model.add(LSTM(128, input_shape=(data.shape), activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10, activation='softmax'))

    opt = tf.keras.optimizers.legacy.Adam(lr=1e-3, decay=1e-5)

    model.compile(loss='sparse_categorical_corossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(data['stars'], data['comment'], epochs=3, validation_data=(testData['stars'], testData['comment']))
