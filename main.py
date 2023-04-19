from dataCleaning import tweetsCleaning, csvCleaning, freqVocab, applyVocab
from utils.trainFT import trainFastText
# from model import text_sentiment_neural_network


class FileNames:
    cleanDataPath = 'data/clean-data'
    dataPath = 'data/data-to-clean/stars_and_comments.csv'
    freqrnc = 'data/data-to-clean/freqrnc2011.csv'
    cleanVocab = 'data/freqVocabClean.csv'
    tweets = ['data/tweets-data/negative.csv', 'data/tweets-data/positive.csv']


if __name__ == '__main__':
    options = [
        {'key': 0, 'name': 'exit'},
        {'key': 1, 'name': 'csvCleaning',
            'func': lambda: csvCleaning(
                data=FileNames.dataPath, 
                clean_data_path=FileNames.cleanDataPath)},
        {'key': 2, 'name': 'tweetsCleaning',
            'func': lambda: tweetsCleaning(
                data=FileNames.tweets, 
                clean_data_path=FileNames.cleanDataPath)},
        {'key': 3, 'name': 'freqVocab',
            'func': lambda: freqVocab(
                data=FileNames.freqrnc, 
                clean_data_path=FileNames.cleanDataPath)},
        {'key': 4, 'name': 'applyVocab',
            'func': lambda: applyVocab(
                commentData=FileNames.cleanDataPath, 
                vocabData=FileNames.cleanVocab)},
        {'key': 5, 'name': 'train FastText',
            'func': lambda: trainFastText(
                commentData=FileNames.cleanDataPath, 
                vocabData=FileNames.cleanVocab)}
    ]
    while True:
        {print(f"{option['key']}: {option['name']}") for option in options}
        try:
            choice = int(input('Choose what to do: '))
            if choice == 0:
                print('Exiting\n')
                break
            options[choice]['func']()
        except IndexError:
            print('No such option\n')
            continue
        except ValueError:
            print('The choice must be a number\n')
            continue

        # text_sentiment_neural_network(
        #     testData=FileNames.cleanTestPath,
        #     trainData=FileNames.cleanTrainPath)
