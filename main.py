from dataCleaning import tweetsCleaning, csvCleaning
from utils.trainFT import trainFastText
# from model import text_sentiment_neural_network


class FileNames:
    cleanDataPath = 'data/clean-data'
    dataPath = 'data/data-to-clean/stars_and_comments.csv'
    freqrnc = 'data/data-to-clean/freqrnc2011.csv'
    tweets = ['data/tweets-data/negative.csv', 'data/tweets-data/positive.csv']
    cleanComments = ['data/clean-data/clean_comments.csv', 'data/clean-data/clean_train_comments.csv', 'data/clean-data/clean_test.comments.csv']
    cleanTweets = ['data/clean-data/tweets-clean/clean_tweets_negative.csv', 'data/clean-data/tweets-clean/clean_tweets_positive.csv']


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
        {'key': 3, 'name': 'train FastText',
            'func': lambda: trainFastText(
                commentData=FileNames.cleanComments, 
                tweetsData=FileNames.cleanTweets)}
        # {'key': 4, 'name': 'RNN model',
        #     'func': lambda: text_sentiment_neural_network(
        #         testData=FileNames.cleanTestPath,
        #         trainData=FileNames.cleanTrainPath)}
    ]
    while True:
        try:
            print('\n')
            {print(f"{option['key']}: {option['name']}") for option in options}
            if (choice := int(input('\nChoose what to do: '))) == 0:
                print('Exiting\n')
                break
            options[choice]['func']()
        except IndexError:
            print('No such option\n')
            continue
        except ValueError:
            print('Exiting\n')
            break
