import pandas as pd
import tensorflow as ts
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


def text_sentiment_neural_network(testData, trainData):
    testData = pd.read_csv(testData)
    trainData = pd.read_csv(trainData)