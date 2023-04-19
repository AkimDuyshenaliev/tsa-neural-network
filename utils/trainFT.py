import pandas as pd
import fasttext
from tqdm import tqdm

from utils.utils import coloring


def trainFastText(commentData, vocabData):
    model = fasttext.train_unsupervised('data/forFastText/tempData.csv', epoch=12, thread=4)
    model.save_model('data/test_model.bin')

    coloring(data='Model trained and saved', r='38', g='05', b='46')


def readFastTextModel(model):
    model = fasttext.load_model(model)
    print(model.words)