import pandas as pd
import fasttext

from keras.datasets import imdb

from utils.utils import coloring


def imdb_data(vocab_size=10000):
    max_features = vocab_size

    (x_train, y_valid), (x_test, y_test_valid) = imdb.load_data(num_words=max_features)

    # Use the default parameters to keras.datasets.imdb.load_data
    start_char = 1
    oov_char = 2
    index_from = 3
    index_from = 3

    # Retrieve the word index file mapping words to indices
    word_index = imdb.get_word_index()

    # Reverse the word index to obtain a dict mapping indices to words
    # And add `index_from` to indices to sync with `x_train`
    inverted_word_index = dict(
        (i + index_from, word) for (word, i) in word_index.items()
    )
    # Update `inverted_word_index` to include `start_char` and `oov_char`
    inverted_word_index[start_char] = ""
    inverted_word_index[oov_char] = ""

    x_train = pd.Series(x_train)
    x_test = pd.Series(x_test)

    # Decode the first sequence in the dataset
    x_valid = x_train.apply(lambda sentence: " ".join(inverted_word_index[i] for i in sentence))
    x_test_valid = x_test.apply(lambda sentence: " ".join(inverted_word_index[i] for i in sentence))

    return (x_valid, y_valid), (x_test_valid, y_test_valid)


def trainFastText(data):
    while True:
        try:
            print('''
0: Quit
1: Train supervised
2: Train unsupervised
            ''')
            if (choice := int(input('What to do: '))) == 0:
                print('Quitting')
                break
            elif choice == 1:
                model = fasttext.train_supervised('data/ft_training_data.txt', epoch=25, wordNgrams=2)
                model.save_model('data/supervised_ft_model.bin')
            elif choice == 2:
                filePath='data/ft_imdb_training_df.txt'
                modelPath='data/ft_imdb_unsupervised.bin'
                (x_valid, y_valid), (x_test_valid, y_test_valid) = imdb_data()
                x_valid = pd.concat([x_valid, x_test_valid], axis=0)
                x_valid.to_csv(filePath)
                
                model = fasttext.train_unsupervised(filePath, epoch=4, thread=4)
                model.save_model(modelPath)
        except:
            print('No such option, quitting')
            break

    coloring(data='Model trained and saved', r='38', g='05', b='46')


def readFastTextModel(model):
    model = fasttext.load_model(model)
    while True:
        print(model.predict(str(input('Write a sentence: '))))