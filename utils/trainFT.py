import pandas as pd
import fasttext

from utils.utils import coloring


def trainFastText(data):
    # model = fasttext.train_unsupervised(data, epoch=4, thread=4)
    model = fasttext.train_supervised('data/ft_training_data.txt', epoch=25, wordNgrams=2)
    model.save_model('data/supervised_ft_model.bin')

    coloring(data='Model trained and saved', r='38', g='05', b='46')


def readFastTextModel(model):
    model = fasttext.load_model(model)
    while True:
        print(model.predict(str(input('Write a sentence: '))))