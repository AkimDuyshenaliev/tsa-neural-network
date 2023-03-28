import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def text_sentiment_neural_network(testData, trainData):
    testData = pd.read_csv(testData)
    trainData = pd.read_csv(trainData)
