import pandas as pd
import fasttext
from tqdm import tqdm

from utils.utils import coloring


def trainFastText(data):
    model = fasttext.train_unsupervised(data, epoch=4, thread=4)
    model.save_model('data/test_model.bin')

    coloring(data='Model trained and saved', r='38', g='05', b='46')


def readFastTextModel(model):
    model = fasttext.load_model(model)
    print(model.get_nearest_neighbors(str(input('Write a word: '))))