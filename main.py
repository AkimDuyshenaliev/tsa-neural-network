from dataCleaning import tweetsCleaning, csvCleaning, makeTrainSeries
from utils.trainFT import trainFastText, readFastTextModel
# from model import text_sentiment_neural_network


class FileNames:
    cleanDataPath = 'data/clean-data'
    dataPath = 'data/data-to-clean/stars_and_comments.csv'
    freqrnc = 'data/data-to-clean/freqrnc2011.csv'
    ftModel = 'data/test_model.bin'
    unsupervisedTrainData = 'data/train-data/unsupervised_train_data.csv'
    tweets = ['data/tweets-data/negative.csv', 'data/tweets-data/positive.csv']
    cleanComments = ['data/clean-data/clean_comments.csv', 'data/clean-data/clean_train_comments.csv', 'data/clean-data/clean_test_comments.csv']
    cleanTweets = ['data/clean-data/tweets-clean/clean_tweets_negative.csv', 'data/clean-data/tweets-clean/clean_tweets_positive.csv']


if __name__ == '__main__':
    options = [
        {'name': 'exit'},
        {'name': 'csvCleaning',
         'func': lambda: csvCleaning(
            data=FileNames.dataPath, 
            clean_data_path=FileNames.cleanDataPath)},
        {'name': 'tweetsCleaning',
         'func': lambda: tweetsCleaning(
            data=FileNames.tweets, 
            clean_data_path=FileNames.cleanDataPath)},
        {'name': 'makeTrainSeries',
         'func': lambda: makeTrainSeries(
            commentData=FileNames.cleanComments, 
            tweetsData=FileNames.cleanTweets)},
        {'name': 'train FastText',
         'func': lambda: trainFastText(
            data=FileNames.unsupervisedTrainData)},
        {'name': 'load FastText model',
         'func': lambda: readFastTextModel(
            model=FileNames.ftModel)}
        # {'name': 'RNN model',
        #  'func': lambda: text_sentiment_neural_network(
        #         testData=FileNames.cleanTestPath,
        #         trainData=FileNames.cleanTrainPath)}
    ]
    while True:
        try:
            print('\n')
            {print(f"{key}: {option['name']}") for key, option in enumerate(options)}
            if (choice := int(input('\nChoose what to do: '))) == 0:
                print('Exiting\n')
                break
        except ValueError:
            print('Exiting\n')
            break
        except IndexError:
            print('No such option\n')
            continue
        options[choice]['func']()

